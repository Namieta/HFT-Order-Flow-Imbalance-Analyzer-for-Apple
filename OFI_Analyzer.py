import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import statsmodels.api as sm
import argparse
import seaborn as sns
import logging
import os
import functools
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ofi_analysis.log'
)

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

DataFrame = pd.DataFrame
Series = pd.Series
Array = np.ndarray

class OFIAnalyzer:
    """Main class for Order Flow Imbalance (OFI) analysis."""
    
    def __init__(self, file_path: str, time_interval_seconds: int = 30, levels: int = 5, cv: int = 5):
        """
        Parameters:
        -----------
        file_path : str
        time_interval_seconds : int
        levels : int
            Number of order book levels to consider
        cv : int
            Number of folds for cross-validation in LASSO
        """
        self.file_path = file_path
        self.time_interval_seconds = time_interval_seconds
        self.levels = levels
        self.cv = cv
        self.df = None
        self.ofi_df = None
        self.multi_level_ofi_df = None
        self.integrated_ofi_df = None
        self.pca_components = None
        self.combined_df = None
        self.results = {}
        
    def load_data(self) -> DataFrame:
        """Load and preprocess the order book data."""
        
        try:
            df = pd.read_csv(self.file_path)
            
            # Convert timestamp columns
            for col in ['ts_event', 'ts_recv']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Convert symbol to category for memory efficiency
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].astype('category')
            
            # Sort by timestamp and sequence
            if 'sequence' in df.columns:
                df = df.sort_values(['ts_event', 'sequence'])
            else:
                df = df.sort_values('ts_event')
            
            self.df = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    @staticmethod
    def _process_level_data(df: DataFrame, level: int) -> DataFrame:
        """
        Process data for a single order book level.
        
        Parameters:
        -----------
        df : DataFrame
            Order book data for a single symbol
        level : int
            Level to process (0-based)
        
        Returns:
        --------
        DataFrame
            Processed DataFrame with order flow calculations for this level
        """
        # Column names for current level
        bid_size_col = f'bid_sz_{level:02d}'
        ask_size_col = f'ask_sz_{level:02d}'
        bid_price_col = f'bid_px_{level:02d}'
        ask_price_col = f'ask_px_{level:02d}'

        logger.debug(f"Processing level {level} data")
        
        # Calculate previous values efficiently
        df[f'{bid_size_col}_prev'] = df[bid_size_col].shift(1)
        df[f'{ask_size_col}_prev'] = df[ask_size_col].shift(1)
        df[f'{bid_price_col}_prev'] = df[bid_price_col].shift(1)
        df[f'{ask_price_col}_prev'] = df[ask_price_col].shift(1)
        
        # Calculate price level changes
        df[f'price_level_bid_{level}_changed'] = df[bid_price_col] != df[f'{bid_price_col}_prev']
        df[f'price_level_ask_{level}_changed'] = df[ask_price_col] != df[f'{ask_price_col}_prev']
        
        # Calculate order flow
        of_bid_col = f'OF{level+1}_b'
        of_ask_col = f'OF{level+1}_a'
        
        # Vectorized conditional calculation
        df[of_bid_col] = np.where(
            df[f'price_level_bid_{level}_changed'],
            df[bid_size_col],
            df[bid_size_col] - df[f'{bid_size_col}_prev']
        )
        
        df[of_ask_col] = np.where(
            df[f'price_level_ask_{level}_changed'],
            df[ask_size_col],
            df[ask_size_col] - df[f'{ask_size_col}_prev']
        )
        
        logger.debug(f"Order flow stats at level {level}:")
        logger.debug(f"Bid OF - Mean: {df[of_bid_col].mean()}, Min: {df[of_bid_col].min()}, Max: {df[of_bid_col].max()}")
        logger.debug(f"Ask OF - Mean: {df[of_ask_col].mean()}, Min: {df[of_ask_col].min()}, Max: {df[of_ask_col].max()}")
    
        return df
    
    def calculate_best_level_ofi(self) -> DataFrame:
        """
        Calculate the best-level OFI according to equation (1).
                
        Returns:
        --------
        DataFrame
            DataFrame with basic OFI values
        """
        if self.df is None:
            self.load_data()
            
        df = self.df
        time_interval_seconds = self.time_interval_seconds
        
        logger.info(f"Calculating BEST_LEVEL OFI with {time_interval_seconds} second intervals")
        
        # Group data by symbol for parallel processing
        symbols = df['symbol'].unique()
        
        # Define a function for processing each symbol in parallel
        def process_symbol(symbol):
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Process level 0 (best level)
            symbol_df = self._process_level_data(symbol_df, 0)
            
            # Define time buckets for aggregation using efficient timestamp conversion
            symbol_df['time_bucket'] = pd.to_datetime(
                (symbol_df['ts_event'].astype(np.int64) // 
                (time_interval_seconds * 1_000_000_000)) * time_interval_seconds * 1_000_000_000
            )
            
            buckets = symbol_df.groupby('time_bucket').size()
            logger.debug(f"Created {len(buckets)} time buckets. Events per bucket - Min: {buckets.min()}, Max: {buckets.max()}, Mean: {buckets.mean():.2f}")

            # Group by time bucket and calculate OFI efficiently
            grouped = symbol_df.groupby('time_bucket')
            
            # Create result dataframe
            ofi_df = pd.DataFrame({
                'OFI1': grouped.apply(lambda x: (x['OF1_b'] - x['OF1_a']).sum()),
                'mid_price_start': grouped.apply(lambda x: (x['bid_px_00'].iloc[0] + x['ask_px_00'].iloc[0]) / 2),
                'mid_price_end': grouped.apply(lambda x: (x['bid_px_00'].iloc[-1] + x['ask_px_00'].iloc[-1]) / 2),
                'event_count': grouped.size(),
                'symbol': symbol
            }).reset_index()
            
            # Calculate returns
            ofi_df['return'] = ofi_df['mid_price_end'] / ofi_df['mid_price_start'] - 1
            
            # Normalize OFI
            ofi_df['OFI1_normalized'] = ofi_df['OFI1'] / ofi_df['event_count']

            logger.debug(f"OFI stats for {symbol}:")
            logger.debug(f"Raw OFI - Mean: {ofi_df['OFI1'].mean():.4f}, Min: {ofi_df['OFI1'].min():.4f}, Max: {ofi_df['OFI1'].max():.4f}")
            logger.debug(f"Norm OFI - Mean: {ofi_df['OFI1_normalized'].mean():.4f}, Min: {ofi_df['OFI1_normalized'].min():.4f}, Max: {ofi_df['OFI1_normalized'].max():.4f}")
            logger.debug(f"Returns - Mean: {ofi_df['return'].mean():.6f}, Min: {ofi_df['return'].min():.6f}, Max: {ofi_df['return'].max():.6f}")
                
            return ofi_df
        
        # Use multiprocessing for parallel processing if there are multiple symbols
        if len(symbols) > 1 and cpu_count() > 1:
            with Pool(processes=min(cpu_count(), len(symbols))) as pool:
                results = pool.map(process_symbol, symbols)
        else:
            results = [process_symbol(symbol) for symbol in symbols]
        
        # Combine results
        if len(results) > 1:
            ofi_df = pd.concat(results, ignore_index=True)
        else:
            ofi_df = results[0]
        
        self.ofi_df = ofi_df
        logger.info(f"Calculated best-level OFI for {len(ofi_df)} time buckets")
        
        return ofi_df
    
    def calculate_multi_level_ofi(self) -> DataFrame:
        """
        Calculate multi-level OFI for several price levels in the limit order book.
        
        Returns:
        --------
        DataFrame
            DataFrame with multi-level OFI values
        """
        if self.df is None:
            self.load_data()
            
        df = self.df
        time_interval_seconds = self.time_interval_seconds
        levels = self.levels
        
        logger.info(f"Calculating multi-level OFI with {time_interval_seconds} second intervals for {levels} levels")
        
        # Group data by symbol for parallel processing
        symbols = df['symbol'].unique()
        
        # Define a function for processing each symbol in parallel
        def process_symbol(symbol):
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Process all levels in parallel
            for level in range(levels):
                symbol_df = self._process_level_data(symbol_df, level)
            
            # Fill NaN values in the first row
            first_idx = symbol_df.index[0]
            for level in range(levels):
                for col_type in ['bid_sz', 'ask_sz', 'bid_px', 'ask_px']:
                    col = f'{col_type}_{level:02d}'
                    prev_col = f'{col}_prev'
                    if col in symbol_df.columns and prev_col in symbol_df.columns:
                        symbol_df.loc[first_idx, prev_col] = symbol_df.loc[first_idx, col]
            
            # Define time buckets
            symbol_df['time_bucket'] = pd.to_datetime(
                (symbol_df['ts_event'].astype(np.int64) // 
                (time_interval_seconds * 1_000_000_000)) * time_interval_seconds * 1_000_000_000
            )
            
            # Prepare aggregation columns
            agg_dict = {
                'bid_px_00': ['first', 'last'],
                'ask_px_00': ['first', 'last'],
            }
            
            # Add OFI columns for each level to the aggregation dictionary
            for level in range(levels):
                of_bid_col = f'OF{level+1}_b'
                of_ask_col = f'OF{level+1}_a'
                if of_bid_col in symbol_df.columns and of_ask_col in symbol_df.columns:
                    agg_dict[of_bid_col] = ['sum']
                    agg_dict[of_ask_col] = ['sum']
            
            # Select only necessary columns for aggregation
            needed_cols = list(agg_dict.keys()) + ['time_bucket']
            reduced_df = symbol_df[needed_cols]
            
            # Group by time bucket and aggregate
            agg_df = reduced_df.groupby('time_bucket').agg(agg_dict)
            
            # Flatten the multi-index columns
            agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
            
            # Calculate mid price and returns
            agg_df['mid_price_start'] = (agg_df['bid_px_00_first'] + agg_df['ask_px_00_first']) / 2
            agg_df['mid_price_end'] = (agg_df['bid_px_00_last'] + agg_df['ask_px_00_last']) / 2
            agg_df['return'] = agg_df['mid_price_end'] / agg_df['mid_price_start'] - 1
            
            # Calculate OFI for each level
            for level in range(levels):
                bid_col = f'OF{level+1}_b_sum'
                ask_col = f'OF{level+1}_a_sum'
                if bid_col in agg_df.columns and ask_col in agg_df.columns:
                    ofi_col = f'OFI{level+1}'
                    agg_df[ofi_col] = agg_df[bid_col] - agg_df[ask_col]
            
            # Count number of events
            event_counts = symbol_df.groupby('time_bucket').size()
            agg_df['event_count'] = event_counts
            
            # Normalize OFIs
            for level in range(levels):
                ofi_col = f'OFI{level+1}'
                if ofi_col in agg_df.columns:
                    agg_df[f'{ofi_col}_normalized'] = agg_df[ofi_col] / agg_df['event_count']
            
            # Add symbol column
            agg_df['symbol'] = symbol
            
            # Reset index to turn time_bucket into a column
            agg_df = agg_df.reset_index()
            
            return agg_df
        
        # Use multiprocessing for parallel processing if there are multiple symbols
        if len(symbols) > 1 and cpu_count() > 1:
            with Pool(processes=min(cpu_count(), len(symbols))) as pool:
                results = pool.map(process_symbol, symbols)
        else:
            results = [process_symbol(symbol) for symbol in tqdm(symbols, desc="Processing symbols")]
        
        # Combine results
        if len(results) > 1:
            multi_level_ofi_df = pd.concat(results, ignore_index=True)
        else:
            multi_level_ofi_df = results[0]
        
        self.multi_level_ofi_df = multi_level_ofi_df
        logger.info(f"Calculated multi-level OFI for {len(multi_level_ofi_df)} time buckets")
        
        return multi_level_ofi_df
    
    def calculate_integrated_ofi(self, training_percentage: float = 0.5) -> Tuple[DataFrame, Dict]:
        """
        Calculate integrated OFI using PCA as per equation (4).
        
        Parameters:
        -----------
        training_percentage : float
            Percentage of data to use for training the PCA model
        
        Returns:
        --------
        tuple
            (DataFrame with integrated OFI, Dictionary with PCA components)
        """
        if self.multi_level_ofi_df is None:
            self.calculate_multi_level_ofi()
            
        multi_level_ofi_df = self.multi_level_ofi_df
        levels = self.levels
        
        logger.info("Calculating integrated OFI using PCA")
        
        symbols = multi_level_ofi_df['symbol'].unique()
        results = []
        pca_components = {}
        
        for symbol in tqdm(symbols, desc="Calculating integrated OFI"):
            symbol_df = multi_level_ofi_df[multi_level_ofi_df['symbol'] == symbol].copy()
            
            # Get the normalized OFI columns
            ofi_cols = [f'OFI{level+1}_normalized' for level in range(levels)]
            existing_cols = [col for col in ofi_cols if col in symbol_df.columns]
            
            if len(existing_cols) < 2:
                logger.warning(f"Not enough OFI levels for symbol {symbol}. "
                            f"Found only {len(existing_cols)} levels. At least 2 levels needed for PCA.")
                
                # Add the symbol dataframe as is
                results.append(symbol_df)
                continue
            
            # Split data into training and testing sets
            n_train = int(len(symbol_df) * training_percentage)
            train_df = symbol_df.iloc[:n_train]
            
            # Extract OFI matrix for PCA
            X_train = train_df[existing_cols].values
            X_train = np.nan_to_num(X_train)
            
            # Standardize data
            X_mean = np.mean(X_train, axis=0)
            X_std = np.std(X_train, axis=0)
            X_std[X_std == 0] = 1  
            X_train_standardized = (X_train - X_mean) / X_std
            
            # Apply PCA to get the first principal component 
            pca = PCA(n_components=1)
            pca.fit(X_train_standardized) 
            
            # Get the first principal vector
            w1 = pca.components_[0]
            
            # Calculate L1 norm of w1
            w1_l1_norm = np.sum(np.abs(w1))
            
            # Normalize weights so they sum to 1 (L1 normalization)
            w1_normalized = w1 / w1_l1_norm
            
            # Store the PCA components for the symbol
            pca_components[symbol] = {
                'w1': w1,
                'w1_normalized': w1_normalized,
                'explained_variance_ratio': pca.explained_variance_ratio_[0]
            }
            
            # Log the percentage of variance explained by the first component
            explained_variance_ratio = pca.explained_variance_ratio_[0]
            logger.info(f"Symbol {symbol}: First principal component explains {explained_variance_ratio:.2%} of variance")
            logger.info(f"Normalized weights for {symbol}: {w1_normalized}")
            
            # Calculate integrated OFI for all data
            X_all = symbol_df[existing_cols].values
            X_all = np.nan_to_num(X_all)
            
            # Standardize all data using same parameters as training
            X_all_standardized = (X_all - X_mean) / X_std
            
            # Apply equation (4) using standardized data
            integrated_ofi = np.dot(X_all_standardized, w1_normalized)
            
            # Add integrated OFI to dataframe
            symbol_df['OFI_integrated'] = integrated_ofi
            
            results.append(symbol_df)
            
        # Combine results
        if len(results) > 1:
            integrated_ofi_df = pd.concat(results, ignore_index=True)
        else:
            integrated_ofi_df = results[0]
        
        self.integrated_ofi_df = integrated_ofi_df
        self.pca_components = pca_components
        
        return integrated_ofi_df, pca_components
    
    def estimate_price_impact_models(self) -> Dict:
        """
        Estimate both price impact models:
        1. PI[1]: Using best-level OFI
        2. PI[I]: Using integrated OFI
        
        Returns:
        --------
        dict
            Dictionary with price impact model results
        """
        if self.integrated_ofi_df is None:
            self.calculate_integrated_ofi()
            
        ofi_df = self.integrated_ofi_df
        
        logger.info("Estimating price impact models")
        
        symbols = ofi_df['symbol'].unique()
        results = {}
        
        for symbol in symbols:
            symbol_ofi = ofi_df[ofi_df['symbol'] == symbol]
            
            # Store results for the symbol
            symbol_results = {}
            
            # PI[1]: Using best-level OFI
            if 'OFI1_normalized' in symbol_ofi.columns:
                X1 = sm.add_constant(symbol_ofi['OFI1_normalized'])
                y = symbol_ofi['return']
                
                model1 = sm.OLS(y, X1)
                results1 = model1.fit()
                
                symbol_results['PI1'] = {
                    'alpha': results1.params[0],
                    'beta': results1.params[1],
                    'r_squared': results1.rsquared,
                    'p_value': results1.pvalues[1],
                    'model_summary': results1.summary()
                }
                
                logger.info(f"PI[1] for {symbol}: R-squared = {results1.rsquared:.4f}, Beta = {results1.params[1]:.8f}")
                logger.info(f"PI[1] for {symbol}: Alpha = {results1.params[0]:.8f}, p-value = {results1.pvalues[1]:.8f}")
            
            # PI[I]: Using integrated OFI
            if 'OFI_integrated' in symbol_ofi.columns:
                XI = sm.add_constant(symbol_ofi['OFI_integrated'])
                y = symbol_ofi['return']
                
                modelI = sm.OLS(y, XI)
                resultsI = modelI.fit()
                
                symbol_results['PII'] = {
                    'alpha': resultsI.params[0],
                    'beta': resultsI.params[1],
                    'r_squared': resultsI.rsquared,
                    'p_value': resultsI.pvalues[1],
                    'model_summary': resultsI.summary()
                }
                
                logger.info(f"PI[I] for {symbol}: R-squared = {resultsI.rsquared:.4f}, Beta = {resultsI.params[1]:.8f}")
                logger.info(f"PI[I] for {symbol}: Alpha = {resultsI.params[0]:.8f}, p-value = {resultsI.pvalues[1]:.8f}")
            
            results[symbol] = symbol_results
        
        self.results['price_impact_models'] = results
        return results
    
    def prepare_cross_asset_ofi_data(self) -> Optional[DataFrame]:
        """
        Prepare cross-asset OFI data for cross-impact analysis.
        This function pivots the data to create a unified time series
        with OFI values for all symbols at each time bucket.
        
        Returns:
        --------
        DataFrame or None
            DataFrame with cross-asset OFI data, or None if only one symbol is available
        """
        if self.integrated_ofi_df is None:
            self.calculate_integrated_ofi()
            
        ofi_df = self.integrated_ofi_df
        
        # First, check if we have multiple symbols
        symbols = ofi_df['symbol'].unique()
        if len(symbols) <= 1:
            logger.warning("Only one symbol detected, cannot perform cross-asset analysis")
            return None
        
        logger.info("Preparing cross-asset OFI data")
        
        # Create pivot for OFI1_normalized (best-level OFI)
        ofi1_pivot = pd.pivot_table(
            ofi_df,
            index='time_bucket', 
            columns='symbol', 
            values='OFI1_normalized',
            aggfunc='first'
        )
        ofi1_pivot.columns = [f'OFI1_{col}' for col in ofi1_pivot.columns]
        
        # Create pivot for OFI_integrated if available
        if 'OFI_integrated' in ofi_df.columns:
            ofi_int_pivot = pd.pivot_table(
                ofi_df,
                index='time_bucket', 
                columns='symbol', 
                values='OFI_integrated',
                aggfunc='first'
            )
            ofi_int_pivot.columns = [f'OFI_int_{col}' for col in ofi_int_pivot.columns]
            
            # Combine the pivots
            ofi_combined = pd.concat([ofi1_pivot, ofi_int_pivot], axis=1)
        else:
            ofi_combined = ofi1_pivot
        
        # Create pivot for returns
        returns_pivot = pd.pivot_table(
            ofi_df,
            index='time_bucket', 
            columns='symbol', 
            values='return',
            aggfunc='first'
        )
        returns_pivot.columns = [f'return_{col}' for col in returns_pivot.columns]
        
        # Combine OFI and returns
        combined_df = pd.concat([ofi_combined, returns_pivot], axis=1)
        
        # Handle NaN values and drop rows with missing values
        combined_df = combined_df.dropna()
        
        self.combined_df = combined_df
        logger.info(f"Created cross-asset OFI data with {len(combined_df)} time points")
        
        return combined_df
    
    def estimate_cross_impact_lasso(self, use_integrated: bool = False) -> Dict:
        """
        Estimate cross-impact models using LASSO regression.
        
        Parameters:
        -----------
        use_integrated : bool
            Whether to use integrated OFI (True) or best-level OFI (False)
        
        Returns:
        --------
        dict
            Dictionary with LASSO model results
        """
        if self.combined_df is None:
            combined_df = self.prepare_cross_asset_ofi_data()
            if combined_df is None:
                logger.error("Cannot estimate cross-impact, combined data is None")
                return {}
        else:
            combined_df = self.combined_df
            
        cv = self.cv
        symbols = self.integrated_ofi_df['symbol'].unique()
        max_iter = 10000
        
        prefix = 'OFI_int_' if use_integrated else 'OFI1_'
        model_type = 'CII' if use_integrated else 'CI1'
        
        logger.info(f"Estimating {model_type} cross-impact models using LASSO")
        
        # Create TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Dictionary to store results
        lasso_results = {}
        
        for symbol in tqdm(symbols, desc=f"Estimating {model_type}"):
            logger.info(f"Estimating {model_type} for {symbol}...")
            
            # Target variable: returns for this symbol
            y_col = f'return_{symbol}'
            if y_col not in combined_df.columns:
                logger.warning(f"No return data for {symbol}, skipping")
                continue
            
            # Feature columns: OFI for all symbols
            X_cols = [col for col in combined_df.columns if col.startswith(prefix)]
            
            if not X_cols:
                logger.warning(f"No {prefix} columns found in data, skipping")
                continue
            
            # Extract features and target
            X = combined_df[X_cols].values
            y = combined_df[y_col].values
            
            try:
                # Use LassoCV to find the optimal alpha parameter efficiently
                lasso_cv = LassoCV(cv=tscv, max_iter=max_iter, random_state=42, n_jobs=-1)
                lasso_cv.fit(X, y)
                
                optimal_alpha = lasso_cv.alpha_
                logger.info(f"Optimal alpha for {symbol}: {optimal_alpha:.6f}")
                
                # Fit final LASSO model with optimal alpha
                lasso_model = Lasso(alpha=optimal_alpha, max_iter=max_iter, random_state=42)
                lasso_model.fit(X, y)
                
                # Calculate metrics
                y_pred = lasso_model.predict(X)
                r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
                
                # Extract coefficients
                coef_dict = dict(zip(X_cols, lasso_model.coef_))
                
                # Separate self-impact from cross-impact
                self_impact_col = f"{prefix}{symbol}"
                
                self_impact = coef_dict.get(self_impact_col, 0)
                cross_impacts = {col.replace(prefix, ''): coef for col, coef in coef_dict.items() 
                                if col != self_impact_col and abs(coef) > 1e-6}
                
                # Store results
                lasso_results[symbol] = {
                    'alpha': lasso_model.intercept_,
                    'self_impact': self_impact,
                    'cross_impacts': cross_impacts,
                    'r_squared': r_squared,
                    'optimal_alpha': optimal_alpha,
                    'model': lasso_model,
                    'feature_names': X_cols
                }
                
                logger.info(f"R-squared for {symbol}: {r_squared:.4f}")
                logger.info(f"Number of non-zero cross-impacts: {len(cross_impacts)}")
                
            except Exception as e:
                logger.error(f"Error estimating LASSO for {symbol}: {str(e)}")
                continue
        
        # Store results
        key = 'cii_results' if use_integrated else 'ci1_results'
        self.results[key] = lasso_results
        
        return lasso_results
    
    def analyze_cross_impact(self, lasso_results: Dict) -> Dict:
        """
        Analyze cross-impact results and provide summary statistics.
        
        Parameters:
        -----------
        lasso_results : dict
            Dictionary with LASSO model results
        
        Returns:
        --------
        dict
            Dictionary with summary statistics
        """
        if not lasso_results:
            logger.warning("Empty lasso_results, cannot analyze cross-impact")
            return {}
            
        # Prepare lists to store statistics
        r_squared_values = []
        self_impact_values = []
        cross_impact_counts = []
        cross_impact_magnitudes = []
        cross_impact_proportions = []
        
        for symbol, results in lasso_results.items():
            r_squared_values.append(results['r_squared'])
            self_impact_values.append(results['self_impact'])
            
            cross_impacts = results['cross_impacts']
            cross_impact_counts.append(len(cross_impacts))
            
            # Calculate magnitude of cross-impacts
            cross_impact_magnitude = sum(abs(v) for v in cross_impacts.values())
            cross_impact_magnitudes.append(cross_impact_magnitude)
            
            # Calculate proportion of total impact from cross-assets
            total_impact = abs(results['self_impact']) + cross_impact_magnitude
            if total_impact > 0:
                cross_impact_proportion = cross_impact_magnitude / total_impact
            else:
                cross_impact_proportion = 0
            cross_impact_proportions.append(cross_impact_proportion)
        
        # Create summary statistics
        summary = {
            'r_squared': {
                'mean': np.mean(r_squared_values),
                'median': np.median(r_squared_values),
                'min': np.min(r_squared_values),
                'max': np.max(r_squared_values),
                'std': np.std(r_squared_values)
            },
            'self_impact': {
                'mean': np.mean(self_impact_values),
                'median': np.median(self_impact_values),
                'min': np.min(self_impact_values),
                'max': np.max(self_impact_values),
                'std': np.std(self_impact_values)
            },
            'cross_impact_count': {
                'mean': np.mean(cross_impact_counts),
                'median': np.median(cross_impact_counts),
                'min': np.min(cross_impact_counts),
                'max': np.max(cross_impact_counts),
                'std': np.std(cross_impact_counts)
            },
            'cross_impact_magnitude': {
                'mean': np.mean(cross_impact_magnitudes),
                'median': np.median(cross_impact_magnitudes),
                'min': np.min(cross_impact_magnitudes),
                'max': np.max(cross_impact_magnitudes),
                'std': np.std(cross_impact_magnitudes)
            },
            'cross_impact_proportion': {
                'mean': np.mean(cross_impact_proportions),
                'median': np.median(cross_impact_proportions),
                'min': np.min(cross_impact_proportions),
                'max': np.max(cross_impact_proportions),
                'std': np.std(cross_impact_proportions)
            }
        }
        
        return summary
    
    def visualize_cross_impact_network(self, lasso_results: Dict, threshold: float = 0.01, 
                                       save_path: str = 'cross_impact_network.png') -> plt.Figure:
        """
        Visualize the cross-impact network.
        
        Parameters:
        -----------
        lasso_results : dict
            Dictionary with LASSO model results
        threshold : float
            Threshold for including cross-impacts in the visualization
        save_path : str
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with the visualization
        """
        if not lasso_results:
            logger.warning("Empty lasso_results, cannot visualize cross-impact network")
            return None
            
        # Prepare data for network visualization
        symbols = list(lasso_results.keys())
        n = len(symbols)
        
        # Create adjacency matrix
        adj_matrix = np.zeros((n, n))
        
        for i, symbol_i in enumerate(symbols):
            results = lasso_results[symbol_i]
            
            # Self-impact on the diagonal
            adj_matrix[i, i] = results['self_impact']
            
            # Cross-impacts for off-diagonal elements
            for symbol_j, impact in results['cross_impacts'].items():
                if symbol_j in symbols:
                    j = symbols.index(symbol_j)
                    adj_matrix[i, j] = impact
        
        # Threshold the matrix
        adj_matrix[np.abs(adj_matrix) < threshold] = 0
        
        # Create a mask for zero values
        mask = (adj_matrix == 0)
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(adj_matrix, mask=mask, cmap='coolwarm', center=0, 
                    xticklabels=symbols, yticklabels=symbols)
        plt.title('Cross-Impact Network (thresholded)')
        plt.xlabel('Source Symbol (j)')
        plt.ylabel('Target Symbol (i)')
        plt.tight_layout()
        plt.savefig(save_path)
        
        logger.info(f"Cross-impact network visualization saved to {save_path}")
        return plt.gcf()
    
    @staticmethod
    def plot_ofi_and_returns(ofi_df: DataFrame, symbol: Optional[str] = None, 
                            save_path: str = 'ofi_returns_plot.png') -> plt.Figure:
        """
        Plot OFI and corresponding returns for visual inspection.
        
        Parameters:
        -----------
        ofi_df : DataFrame
            DataFrame with OFI data
        symbol : str or None
            Symbol to plot (if None, use all data)
        save_path : str
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with the plot
        """
        if symbol:
            plot_df = ofi_df[ofi_df['symbol'] == symbol]
        else:
            plot_df = ofi_df
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot OFI
        ax1.plot(plot_df['time_bucket'], plot_df['OFI1_normalized'], 'b-')
        ax1.set_title(f'Normalized OFI for {symbol if symbol else "all symbols"}')
        ax1.set_ylabel('OFI1 (normalized)')
        ax1.grid(True)
        
        # Plot returns
        ax2.plot(plot_df['time_bucket'], plot_df['return'], 'r-')
        ax2.set_title('Returns')
        ax2.set_ylabel('Return')
        ax2.set_xlabel('Time')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        
        # Relationship between OFI and returns
        plt.figure(figsize=(8, 6))
        plt.scatter(plot_df['OFI1_normalized'], plot_df['return'])
        plt.title(f'OFI vs Returns for {symbol if symbol else "all symbols"}')
        plt.xlabel('OFI1 (normalized)')
        plt.ylabel('Return')
        plt.grid(True)
        
        # Add regression line efficiently
        if len(plot_df) > 1:  
            X = plot_df['OFI1_normalized'].values.reshape(-1, 1)
            y = plot_df['return'].values
            # Handle potential NaNs
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            if mask.sum() > 1:  # Make sure we have at least 2 valid data points
                X_valid = X[mask]
                y_valid = y[mask]
                model = LinearRegression()
                model.fit(X_valid, y_valid)
                
                # Create sorted X values for smoother line
                X_sorted = np.sort(X_valid, axis=0)
                plt.plot(X_sorted, model.predict(X_sorted), color='red', linewidth=2)
        
        plt.savefig('ofi_returns_scatter.png')
        
        return fig
    
    @staticmethod
    def plot_multi_level_ofi(ofi_df: DataFrame, symbol: Optional[str] = None, 
                            max_levels: int = 5,
                            save_path: str = 'multi_level_ofi_plot.png') -> plt.Figure:
        """
        Plot multi-level OFI and integrated OFI.
        
        Parameters:
        -----------
        ofi_df : DataFrame
            DataFrame with multi-level OFI data
        symbol : str or None
            Symbol to plot (if None, use first symbol)
        max_levels : int
            Maximum number of order book levels to consider
        save_path : str
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with the plot
        """
        if symbol:
            plot_df = ofi_df[ofi_df['symbol'] == symbol]
        else:
            # Use the first symbol if none specified
            symbol = ofi_df['symbol'].iloc[0]
            plot_df = ofi_df[ofi_df['symbol'] == symbol]
        
        # Determine available OFI levels
        ofi_cols = [f'OFI{level+1}_normalized' for level in range(max_levels)]
        existing_cols = [col for col in ofi_cols if col in plot_df.columns]
        
        if not existing_cols:
            logger.warning(f"No OFI columns found for {symbol}")
            return None
        
        # Create a figure with subplots
        n_plots = len(existing_cols) + 2  # +1 for integrated OFI, +1 for returns
        fig, axs = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)
        
        # Convert axs to list if there's only one subplot
        if n_plots == 1:
            axs = [axs]
        
        # Plot each OFI level
        for i, col in enumerate(existing_cols):
            level = i + 1
            axs[i].plot(plot_df['time_bucket'], plot_df[col], 'b-')
            axs[i].set_title(f'OFI Level {level} for {symbol}')
            axs[i].set_ylabel(f'OFI{level} (normalized)')
            axs[i].grid(True)
        
        # Plot integrated OFI if available
        if 'OFI_integrated' in plot_df.columns:
            i = len(existing_cols)
            axs[i].plot(plot_df['time_bucket'], plot_df['OFI_integrated'], 'g-')
            axs[i].set_title(f'Integrated OFI for {symbol}')
            axs[i].set_ylabel('OFI Integrated')
            axs[i].grid(True)
        
        # Plot returns
        axs[-1].plot(plot_df['time_bucket'], plot_df['return'], 'r-')
        axs[-1].set_title('Returns')
        axs[-1].set_ylabel('Return')
        axs[-1].set_xlabel('Time')
        axs[-1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        
        # Plot scatter plots for OFI levels vs returns
        n_cols = 2
        n_rows = (len(existing_cols) + 1) // n_cols + 1  # +1 for integrated OFI
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        axs = axs.flatten() if n_rows * n_cols > 1 else [axs]
        
        # Plot each OFI level vs returns
        for i, col in enumerate(existing_cols):
            if i < len(axs):
                level = i + 1
                ax = axs[i]
                ax.scatter(plot_df[col], plot_df['return'])
                ax.set_title(f'OFI Level {level} vs Returns')
                ax.set_xlabel(f'OFI{level} (normalized)')
                ax.set_ylabel('Return')
                ax.grid(True)
                
                # Add regression line
                if len(plot_df) > 1:
                    X = plot_df[col].values.reshape(-1, 1)
                    y = plot_df['return'].values
                    # Handle potential NaNs
                    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                    if mask.sum() > 1:
                        X_valid = X[mask]
                        y_valid = y[mask]
                        model = LinearRegression()
                        model.fit(X_valid, y_valid)
                        
                        # Create sorted X values for smoother line
                        X_sorted = np.sort(X_valid, axis=0)
                        ax.plot(X_sorted, model.predict(X_sorted), color='red', linewidth=2)
        
        # Plot integrated OFI vs returns if available
        if 'OFI_integrated' in plot_df.columns:
            i = len(existing_cols)
            if i < len(axs):
                ax = axs[i]
                ax.scatter(plot_df['OFI_integrated'], plot_df['return'])
                ax.set_title('Integrated OFI vs Returns')
                ax.set_xlabel('OFI Integrated')
                ax.set_ylabel('Return')
                ax.grid(True)
                
                # Add regression line
                if len(plot_df) > 1:
                    X = plot_df['OFI_integrated'].values.reshape(-1, 1)
                    y = plot_df['return'].values
                    # Handle potential NaNs
                    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                    if mask.sum() > 1:
                        X_valid = X[mask]
                        y_valid = y[mask]
                        model = LinearRegression()
                        model.fit(X_valid, y_valid)
                        
                        # Create sorted X values for smoother line
                        X_sorted = np.sort(X_valid, axis=0)
                        ax.plot(X_sorted, model.predict(X_sorted), color='red', linewidth=2)
        
        # Hide any unused subplots
        for i in range(len(existing_cols) + 1, len(axs)):
            axs[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('multi_level_ofi_scatter.png')
        
        return fig

    @staticmethod
    def plot_correlation_matrix(ofi_df: DataFrame, symbol: str, max_levels: int = 5, 
                            save_path: str = 'ofi_correlation_matrix.png') -> plt.Figure:
        """
        Plot correlation matrix between different OFI levels.
        
        Parameters:
        -----------
        ofi_df : DataFrame
            DataFrame with multi-level OFI data
        symbol : str
            Symbol to plot
        max_levels : int
            Maximum number of order book levels to consider
        save_path : str
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with the plot, or None if not enough data
        """
        # Filter data for the specified symbol
        symbol_df = ofi_df[ofi_df['symbol'] == symbol].copy()
        
        # Get all OFI columns
        ofi_cols = [f'OFI{level+1}_normalized' for level in range(max_levels) 
                    if f'OFI{level+1}_normalized' in symbol_df.columns]
        
        # Add integrated OFI if available
        if 'OFI_integrated' in symbol_df.columns:
            ofi_cols.append('OFI_integrated')
        
        # Check if we have enough columns
        if len(ofi_cols) < 2:
            logger.warning(f"Not enough OFI columns for correlation matrix for {symbol}")
            return None
        
        # Calculate correlation matrix
        corr_matrix = symbol_df[ofi_cols].corr()
        
        # Remove mask to show full matrix (not just lower triangle)
        # mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f')
        
        plt.title(f'Correlation Matrix of OFI Levels for {symbol}', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path)
        logger.info(f"Correlation matrix saved to {save_path}")
        return plt.gcf()
        
    def run_basic_analysis(self) -> Dict:
        """
        Run basic OFI analysis:
        1. Calculate OFI
        2. Estimate price impact
        
        Returns:
        --------
        dict
            Dictionary with analysis results
        """
        logger.info("Running basic OFI analysis")
        
        # Calculate OFI
        ofi_df = self.calculate_best_level_ofi()
        
        # Estimate price impact
        pi_results = {}
        symbols = ofi_df['symbol'].unique()
        
        for symbol in symbols:
            symbol_ofi = ofi_df[ofi_df['symbol'] == symbol]
            
            # Prepare data for regression
            X = sm.add_constant(symbol_ofi['OFI1_normalized'])
            y = symbol_ofi['return']
            
            # Estimate model
            model = sm.OLS(y, X)
            results = model.fit()
            
            pi_results[symbol] = {
                'alpha': results.params[0],
                'beta': results.params[1],
                'r_squared': results.rsquared,
                'p_value': results.pvalues[1],
                'model_summary': results.summary()
            }
            
            logger.info(f"Price Impact (PI[1]) for {symbol}: R-squared = {results.rsquared:.4f}")
            logger.info(f"Alpha for {symbol}: {results.params[0]:.8f}")
            logger.info(f"Beta for {symbol}: {results.params[1]:.8f}")
            logger.info(f"p-value for {symbol}: {results.pvalues[1]:.4f}")

        
        # Create first plot
        first_symbol = symbols[0]
        self.plot_ofi_and_returns(ofi_df, symbol=first_symbol)
        
        # Save OFI data to csv
        ofi_df.to_csv('ofi_results.csv', index=False)
        logger.info("Basic analysis results saved to ofi_results.csv")
        
        return {
            'ofi_data': ofi_df,
            'price_impact': pi_results
        }
    
    def run_multi_level_analysis(self) -> Dict:
        """
        Run multi-level OFI analysis:
        1. Calculate multi-level OFI
        2. Calculate integrated OFI
        3. Estimate price impact models
        
        Returns:
        --------
        dict
            Dictionary with analysis results
        """
        logger.info("Running multi-level OFI analysis")
        
        # Calculate multi-level OFI
        multi_level_ofi_df = self.calculate_multi_level_ofi()
        
        # Calculate integrated OFI
        integrated_ofi_df, pca_components = self.calculate_integrated_ofi()
        
        # Estimate price impact models
        pi_results = self.estimate_price_impact_models()
        
        # Get first symbol for plotting
        first_symbol = integrated_ofi_df['symbol'].unique()[0]
        
        # Plot multi-level OFI and integrated OFI
        self.plot_multi_level_ofi(integrated_ofi_df, symbol=first_symbol, max_levels=self.levels)
        
        # Plot correlation matrix
        self.plot_correlation_matrix(integrated_ofi_df, symbol=first_symbol, max_levels=self.levels,
                                   save_path=f'ofi_correlation_matrix_{first_symbol}.png')
        
        # Save OFI data to csv
        integrated_ofi_df.to_csv('multi_level_ofi_results.csv', index=False)
        logger.info("Multi-level analysis results saved to multi_level_ofi_results.csv")
        
        return {
            'multi_level_ofi_df': multi_level_ofi_df,
            'integrated_ofi_df': integrated_ofi_df,
            'pca_components': pca_components,
            'price_impact_models': pi_results
        }
    
    def run_cross_asset_analysis(self) -> Optional[Dict]:
        """
        Run cross-asset OFI analysis:
        1. Prepare cross-asset OFI data
        2. Estimate cross-impact models
        3. Analyze results
        
        Returns:
        --------
        dict or None
            Dictionary with analysis results, or None if cross-asset analysis cannot be performed
        """
        if self.integrated_ofi_df is None:
            self.calculate_integrated_ofi()
            
        # Check if we have multiple symbols
        symbols = self.integrated_ofi_df['symbol'].unique()
        if len(symbols) <= 1:
            logger.warning(f"Only one asset ({symbols[0]}) found in the dataset. "
                         f"Cross-asset analysis requires multiple assets.")
            return None
        
        logger.info("Running cross-asset OFI analysis")
        
        # Prepare cross-asset OFI data
        combined_df = self.prepare_cross_asset_ofi_data()
        if combined_df is None:
            return None
        
        # Estimate cross-impact models using best-level OFI (CI[1])
        logger.info("Estimating cross-impact models using best-level OFI (CI[1])")
        ci1_results = self.estimate_cross_impact_lasso(use_integrated=False)
        
        # Estimate cross-impact models using integrated OFI (CII)
        logger.info("Estimating cross-impact models using integrated OFI (CII)")
        cii_results = self.estimate_cross_impact_lasso(use_integrated=True)
        
        # Analyze cross-impact results
        logger.info("Analyzing cross-impact results")
        ci1_summary = self.analyze_cross_impact(ci1_results)
        cii_summary = self.analyze_cross_impact(cii_results)
        
        # Visualize cross-impact network
        logger.info("Visualizing cross-impact network")
        ci1_network = self.visualize_cross_impact_network(ci1_results, save_path='ci1_network.png')
        cii_network = self.visualize_cross_impact_network(cii_results, save_path='cii_network.png')
        
        # Save results to CSV files
        self.integrated_ofi_df.to_csv('integrated_ofi_results.csv', index=False)
        combined_df.to_csv('cross_asset_ofi_results.csv', index=False)
        
        logger.info("Cross-asset analysis results saved to integrated_ofi_results.csv and cross_asset_ofi_results.csv")
        
        return {
            'multi_level_ofi': self.multi_level_ofi_df,
            'integrated_ofi': self.integrated_ofi_df,
            'combined_df': combined_df,
            'pca_components': self.pca_components,
            'ci1_results': ci1_results,
            'cii_results': cii_results,
            'ci1_summary': ci1_summary,
            'cii_summary': cii_summary,
            'ci1_network': ci1_network, 
            'cii_network': cii_network 
        }
    
    def run_all_analyses(self) -> Dict:
        """
        Run all available analyses.
        
        Returns:
        --------
        dict
            Dictionary with all analysis results
        """
        logger.info("Running ALL analyses")
        
        # 1. Run Basic OFI Analysis
        logger.info("\n\n1. RUNNING BASIC OFI ANALYSIS")
        logger.info("===============================")
        basic_results = self.run_basic_analysis()
        
        # 2. Run Multi-Level OFI Analysis
        logger.info("\n\n2. RUNNING MULTI-LEVEL OFI ANALYSIS")
        logger.info("=====================================")
        multi_results = self.run_multi_level_analysis()
        
        # 3. Check if we can run Cross-Asset OFI Analysis
        symbols = self.integrated_ofi_df['symbol'].unique()
        cross_results = None
        
        if len(symbols) <= 1:
            logger.info("\n\n3. SKIPPING CROSS-ASSET OFI ANALYSIS")
            logger.info("=========================================")
            logger.info(f"Only one asset ({symbols[0]}) found in the dataset.")
            logger.info("Cross-asset analysis requires multiple assets.")
        else:
            # Run cross-asset OFI analysis
            logger.info("\n\n3. RUNNING CROSS-ASSET OFI ANALYSIS")
            logger.info("=====================================")
            cross_results = self.run_cross_asset_analysis()
        
        logger.info("\n\nALL ANALYSES COMPLETE!")
        
        return {
            'basic_results': basic_results,
            'multi_results': multi_results,
            'cross_results': cross_results
        }

def get_user_input_cli() -> Tuple[str, int, int, int, str]:
    """Get parameters from command line arguments."""
    parser = argparse.ArgumentParser(description='Order Flow Imbalance (OFI) Analysis')
    parser.add_argument('file_path', type=str, help='Path to the order book data CSV file')
    parser.add_argument('--time_interval', '-t', type=int, default=30, 
                        help='Time interval in seconds for calculating OFI (default: 30)')
    parser.add_argument('--levels', '-l', type=int, default=5,
                        help='Number of order book levels to consider (default: 5)')
    parser.add_argument('--cv', '-c', type=int, default=5,
                        help='Number of folds for cross-validation in LASSO (default: 5)')
    parser.add_argument('--analysis_type', '-a', type=str, choices=['basic', 'multi', 'cross', 'all'], 
                        default='all', help='Type of analysis to perform (default: all)')
    
    args = parser.parse_args()
    return args.file_path, args.time_interval, args.levels, args.cv, args.analysis_type

def get_user_input_interactive() -> Tuple[str, int, int, int, str]:
    """Get user input interactively when not running from command line."""
    file_path = input("Enter the path to the CSV file: ")
    
    while True:
        try:
            time_interval = int(input("Enter the time interval in seconds (default: 30): ") or "30")
            if time_interval <= 0:
                print("Time interval must be positive. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer for the time interval.")
    
    while True:
        try:
            levels = int(input("Enter the number of order book levels to consider (default: 5): ") or "5")
            if levels <= 0:
                print("Number of levels must be positive. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer for the number of levels.")
    
    while True:
        try:
            cv = int(input("Enter the number of folds for cross-validation in LASSO (default: 5): ") or "5")
            if cv <= 1:
                print("Number of folds must be greater than 1. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer for the number of folds.")
    
    analysis_type = ""
    while analysis_type not in ['basic', 'multi', 'cross', 'all']:
        analysis_type = input("Enter analysis type (basic/multi/cross/all) (default: all): ").lower() or "all"
        if analysis_type not in ['basic', 'multi', 'cross', 'all']:
            print("Please enter a valid analysis type: basic, multi, cross, or all.")
    
    return file_path, time_interval, levels, cv, analysis_type

def main():
    """Main function to run the OFI analysis."""
    import sys
    
    # Configure logging to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # Check if running from command line with arguments
    if len(sys.argv) > 1:
        file_path, time_interval, levels, cv, analysis_type = get_user_input_cli()
    else:
        # If no command line arguments, get input interactively
        file_path, time_interval, levels, cv, analysis_type = get_user_input_interactive()
    
    # Create OFI analyzer
    analyzer = OFIAnalyzer(file_path, time_interval, levels, cv)
    
    # Run analysis based on type
    if analysis_type == 'basic':
        analyzer.run_basic_analysis()
    elif analysis_type == 'multi':
        analyzer.run_multi_level_analysis()
    elif analysis_type == 'cross':
        analyzer.run_cross_asset_analysis()
    else:  # 'all'
        analyzer.run_all_analyses()

if __name__ == "__main__":
    main()