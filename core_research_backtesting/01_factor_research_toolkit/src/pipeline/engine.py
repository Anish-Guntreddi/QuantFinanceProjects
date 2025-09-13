"""Main pipeline engine for factor research"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import importlib
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.loader import DataLoader
from data.universe import UniverseConstructor
from data.point_in_time import PointInTimeJoiner
from factors.base import BaseFactor
from transforms.neutralization import Neutralizer
from transforms.standardization import Standardizer
from analytics.ic_analysis import ICAnalyzer
from analytics.turnover import TurnoverAnalyzer
from analytics.capacity import CapacityAnalyzer
from analytics.attribution import AttributionEngine


class FactorPipeline:
    """Complete pipeline for factor research"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize factor pipeline
        
        Parameters:
        -----------
        config_path : Optional[str]
            Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.data_loader = DataLoader()
        self.universe_constructor = UniverseConstructor()
        self.pit_joiner = PointInTimeJoiner()
        
        self.factors = []
        self.transforms = []
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def add_factor(self, factor: BaseFactor):
        """Add a factor to the pipeline"""
        self.factors.append(factor)
    
    def add_transform(self, transform):
        """Add a transform to the pipeline"""
        self.transforms.append(transform)
    
    def load_factors_from_config(self):
        """Load factors from configuration"""
        if 'factors' not in self.config:
            return
        
        for category, factor_list in self.config['factors'].items():
            for factor_def in factor_list:
                # Dynamic import of factor class
                module_name = f"factors.{category}"
                class_name = factor_def['class']
                
                try:
                    module = importlib.import_module(module_name)
                    factor_class = getattr(module, class_name)
                    
                    # Instantiate factor with parameters
                    params = factor_def.get('params', {})
                    factor = factor_class(**params)
                    
                    self.add_factor(factor)
                except Exception as e:
                    print(f"Warning: Could not load factor {class_name}: {e}")
    
    def run(
        self,
        start_date: str,
        end_date: str,
        universe: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run complete factor pipeline
        
        Parameters:
        -----------
        start_date : str
            Start date for analysis
        end_date : str
            End date for analysis
        universe : Optional[List[str]]
            List of symbols (if None, use S&P 500)
            
        Returns:
        --------
        Tuple[pd.DataFrame, Dict]
            Factor values and analytics results
        """
        # Step 1: Define universe
        if universe is None:
            universe = self.universe_constructor.get_sp500_universe()
        
        print(f"Running pipeline for {len(universe)} symbols from {start_date} to {end_date}")
        
        # Step 2: Load price data
        print("Loading price data...")
        price_data = self.data_loader.load_price_data(universe, start_date, end_date)
        
        # Step 3: Load fundamental data
        print("Loading fundamental data...")
        fundamental_data = self.data_loader.load_fundamental_data(universe, end_date)
        
        # Step 4: Create aligned dataset
        print("Creating aligned dataset...")
        aligned_data = self.pit_joiner.create_aligned_dataset(
            price_data, fundamental_data, start_date, end_date
        )
        
        # Step 5: Calculate returns
        print("Calculating returns...")
        price_df = pd.DataFrame({
            symbol: data['Adj Close'] for symbol, data in price_data.items()
            if 'Adj Close' in data.columns
        })
        
        returns = self.data_loader.calculate_returns(price_df)
        forward_returns = self.data_loader.calculate_forward_returns(price_df)
        
        # Step 6: Calculate factors
        print("Calculating factors...")
        factor_values = {}
        
        if not self.factors:
            # Load from config if no factors added manually
            self.load_factors_from_config()
        
        for factor in self.factors:
            try:
                print(f"  Calculating {factor.name}...")
                
                # Prepare data for factor calculation
                factor_data = self._prepare_factor_data(aligned_data, price_df, returns)
                
                # Calculate factor
                values = factor.calculate_and_validate(factor_data)
                factor_values[factor.name] = values
                
            except Exception as e:
                print(f"  Error calculating {factor.name}: {e}")
                continue
        
        # Convert to DataFrame
        factor_df = pd.DataFrame(factor_values)
        
        # Step 7: Apply transforms
        print("Applying transforms...")
        for transform in self.transforms:
            factor_df = transform.apply(factor_df)
        
        # Step 8: Calculate analytics
        print("Calculating analytics...")
        analytics = self._calculate_analytics(factor_df, forward_returns, price_df)
        
        # Store results
        self.results = {
            'factor_values': factor_df,
            'analytics': analytics
        }
        
        return factor_df, analytics
    
    def _prepare_factor_data(
        self,
        aligned_data: pd.DataFrame,
        price_df: pd.DataFrame,
        returns: Dict
    ) -> pd.DataFrame:
        """Prepare data for factor calculation"""
        # Create a simplified data structure for factors
        # This would need to be customized based on actual data structure
        
        factor_data = pd.DataFrame(index=aligned_data.index)
        
        # Add price data
        if not price_df.empty:
            factor_data['price'] = price_df.mean(axis=1)  # Simple average for demo
        
        # Add returns
        if 1 in returns:
            factor_data['returns'] = returns[1].mean(axis=1)
        
        # Add fundamental data from aligned dataset
        # Extract fundamentals (this is simplified - actual implementation would be more sophisticated)
        for col in aligned_data.columns:
            if 'book_value' in col:
                factor_data['book_value'] = aligned_data[col]
            elif 'market_cap' in col:
                factor_data['market_cap'] = aligned_data[col]
            elif 'earnings' in col:
                factor_data['earnings'] = aligned_data[col]
            elif 'revenue' in col:
                factor_data['revenue'] = aligned_data[col]
            elif 'free_cash_flow' in col or 'fcf' in col:
                factor_data['free_cash_flow'] = aligned_data[col]
        
        return factor_data
    
    def _calculate_analytics(
        self,
        factor_df: pd.DataFrame,
        forward_returns: Dict,
        price_df: pd.DataFrame
    ) -> Dict:
        """Calculate comprehensive analytics"""
        analytics = {}
        
        # IC Analysis
        ic_analyzer = ICAnalyzer()
        if 1 in forward_returns and not forward_returns[1].empty:
            print("  Calculating IC metrics...")
            for factor in factor_df.columns:
                factor_series = factor_df[factor].to_frame()
                ic = ic_analyzer.calculate_ic(
                    factor_series,
                    forward_returns[1],
                    method='spearman'
                )
                analytics[f"{factor}_ic"] = ic.mean()
            
            # IC decay
            if len(forward_returns) > 1:
                ic_decay = ic_analyzer.calculate_ic_decay(
                    factor_df,
                    forward_returns
                )
                analytics['ic_decay'] = ic_decay
        
        # Turnover Analysis
        turnover_analyzer = TurnoverAnalyzer()
        print("  Calculating turnover metrics...")
        for factor in factor_df.columns[:3]:  # Limit to first 3 factors for speed
            factor_turnover = turnover_analyzer.calculate_factor_turnover(
                factor_df[[factor]],
                n_quantiles=5
            )
            analytics[f"{factor}_turnover"] = factor_turnover
        
        # Capacity Analysis (simplified)
        if not price_df.empty:
            capacity_analyzer = CapacityAnalyzer()
            print("  Calculating capacity estimates...")
            # Create mock volume data for demo
            volume_df = pd.DataFrame(
                np.random.uniform(1e6, 1e8, size=price_df.shape),
                index=price_df.index,
                columns=price_df.columns
            )
            
            # Create mock positions for capacity analysis
            positions = factor_df.iloc[:, :min(3, len(factor_df.columns))]
            positions = positions.fillna(0).abs() / positions.abs().sum(axis=1).values.reshape(-1, 1)
            
            capacity = capacity_analyzer.calculate_capacity_via_adv(
                positions,
                volume_df,
                price_df
            )
            analytics['capacity'] = capacity
        
        return analytics
    
    def generate_report(self, output_path: str = "./reports/factor_report.html"):
        """Generate comprehensive factor research report"""
        if not self.results:
            print("No results to report. Run pipeline first.")
            return
        
        report_content = self._create_report_content()
        
        # Save report
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        print(f"Report saved to {output_path}")
    
    def _create_report_content(self) -> str:
        """Create HTML report content"""
        factor_values = self.results.get('factor_values', pd.DataFrame())
        analytics = self.results.get('analytics', {})
        
        html = """
        <html>
        <head>
            <title>Factor Research Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Factor Research Report</h1>
            <h2>Executive Summary</h2>
            <p>Analysis Period: """ + str(factor_values.index[0]) + " to " + str(factor_values.index[-1]) + """</p>
            <p>Number of Factors: """ + str(len(factor_values.columns)) + """</p>
            
            <h2>Factor IC Analysis</h2>
            <table>
                <tr><th>Factor</th><th>Mean IC</th></tr>
        """
        
        for key, value in analytics.items():
            if '_ic' in key and isinstance(value, (int, float)):
                html += f"<tr><td>{key.replace('_ic', '')}</td><td>{value:.4f}</td></tr>"
        
        html += """
            </table>
            
            <h2>Turnover Analysis</h2>
            <table>
                <tr><th>Factor</th><th>Annual Turnover</th><th>Transaction Costs (bps)</th></tr>
        """
        
        for key, value in analytics.items():
            if '_turnover' in key and isinstance(value, dict):
                factor_name = key.replace('_turnover', '')
                annual_turnover = value.get('annual_turnover', 0)
                costs = value.get('transaction_costs', 0) * 10000  # Convert to bps
                html += f"<tr><td>{factor_name}</td><td>{annual_turnover:.1f}</td><td>{costs:.1f}</td></tr>"
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html