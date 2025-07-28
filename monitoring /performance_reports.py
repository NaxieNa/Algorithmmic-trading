"""
Complete Performance Report Generator
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import numpy as np
from datetime import datetime, timedelta
import json

plt.style.use('seaborn-v0_8-darkgrid')

class PerformanceReporter:
    """Comprehensive performance report generator"""
    
    def __init__(self):
        self.report_data = {}
        self.colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
        
    def generate_report(self, trades: List[Dict], equity_curve: List[float],
                       positions: Dict = None, config: Dict = None) -> Dict:
        """Generate comprehensive performance report"""
        
        if not trades or len(equity_curve) < 2:
            return {"error": "Insufficient data for report generation"}
        
        # Calculate all metrics
        metrics = self._calculate_all_metrics(trades, equity_curve)
        
        # Generate visualizations
        self._generate_all_charts(trades, equity_curve, metrics)
        
        # Generate HTML report
        html_report = self._generate_html_report(metrics, trades, config)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"performance_report_{timestamp}.html"
        
        with open(report_filename, 'w') as f:
            f.write(html_report)
        
        return {
            "metrics": metrics,
            "report_file": report_filename,
            "charts_generated": True
        }
    
    def _calculate_all_metrics(self, trades: List[Dict], 
                              equity_curve: List[float]) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Convert to numpy for easier calculation
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Trade analysis
        trade_returns = [t.get('return', 0) for t in trades]
        winning_trades = [t for t in trades if t.get('return', 0) > 0]
        losing_trades = [t for t in trades if t.get('return', 0) <= 0]
        
        # Calculate metrics
        metrics = {
            # Return metrics
            'total_return': (equity[-1] / equity[0] - 1) * 100,
            'cagr': self._calculate_cagr(equity),
            'monthly_return': self._calculate_monthly_returns(returns),
            
            # Risk metrics
            'volatility': returns.std() * np.sqrt(252 * 96) * 100,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(equity),
            'max_drawdown': self._calculate_max_drawdown(equity),
            'max_drawdown_duration': self._calculate_max_dd_duration(equity),
            
            # Trade metrics
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
            'avg_win': np.mean([t['return'] for t in winning_trades]) * 100 if winning_trades else 0,
            'avg_loss': np.mean([t['return'] for t in losing_trades]) * 100 if losing_trades else 0,
            'profit_factor': self._calculate_profit_factor(trades),
            'expectancy': np.mean(trade_returns) * 100 if trade_returns else 0,
            
            # Risk/Reward
            'risk_reward_ratio': abs(
                np.mean([t['return'] for t in winning_trades]) / 
                np.mean([t['return'] for t in losing_trades])
            ) if losing_trades and winning_trades else 0,
            
            # Additional metrics
            'kelly_percentage': self._calculate_kelly_percentage(trades),
            'ulcer_index': self._calculate_ulcer_index(equity),
            'recovery_factor': self._calculate_recovery_factor(equity),
            'payoff_ratio': self._calculate_payoff_ratio(trades),
            
            # Trade duration
            'avg_trade_duration': np.mean([t.get('duration', 0) for t in trades]),
            'avg_win_duration': np.mean([t.get('duration', 0) for t in winning_trades]) if winning_trades else 0,
            'avg_loss_duration': np.mean([t.get('duration', 0) for t in losing_trades]) if losing_trades else 0,
            
            # Consecutive wins/losses
            'max_consecutive_wins': self._calculate_max_consecutive(trade_returns, True),
            'max_consecutive_losses': self._calculate_max_consecutive(trade_returns, False),
            
            # Distribution metrics
            'return_skewness': self._safe_calc(lambda: returns.skew()),
            'return_kurtosis': self._safe_calc(lambda: returns.kurtosis()),
            
            # VaR and CVaR
            'var_95': np.percentile(returns, 5) * 100,
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean() * 100,
        }
        
        return metrics
    
    def _generate_all_charts(self, trades: List[Dict], 
                           equity_curve: List[float],
                           metrics: Dict):
        """Generate all performance charts"""
        
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Equity Curve
        ax1 = plt.subplot(4, 2, 1)
        self._plot_equity_curve(ax1, equity_curve)
        
        # 2. Drawdown
        ax2 = plt.subplot(4, 2, 2)
        self._plot_drawdown(ax2, equity_curve)
        
        # 3. Returns Distribution
        ax3 = plt.subplot(4, 2, 3)
        self._plot_returns_distribution(ax3, trades)
        
        # 4. Monthly Returns Heatmap
        ax4 = plt.subplot(4, 2, 4)
        self._plot_monthly_returns_heatmap(ax4, equity_curve)
        
        # 5. Rolling Sharpe Ratio
        ax5 = plt.subplot(4, 2, 5)
        self._plot_rolling_sharpe(ax5, equity_curve)
        
        # 6. Win/Loss Analysis
        ax6 = plt.subplot(4, 2, 6)
        self._plot_win_loss_analysis(ax6, trades)
        
        # 7. Trade Duration Analysis
        ax7 = plt.subplot(4, 2, 7)
        self._plot_trade_duration(ax7, trades)
        
        # 8. Cumulative Returns by Strategy
        ax8 = plt.subplot(4, 2, 8)
        self._plot_strategy_comparison(ax8, trades)
        
        plt.tight_layout()
        plt.savefig('performance_charts.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_equity_curve(self, ax, equity_curve):
        """Plot equity curve with benchmark"""
        
        # Create time index
        time_index = pd.date_range(
            end=datetime.now(),
            periods=len(equity_curve),
            freq='15min'
        )
        
        # Plot equity curve
        ax.plot(time_index, equity_curve, 'b-', linewidth=2, label='Portfolio')
        
        # Add starting capital line
        ax.axhline(y=equity_curve[0], color='gray', linestyle='--', alpha=0.5)
        
        # Format
        ax.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics text
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        ax.text(0.02, 0.98, f'Total Return: {total_return:.2f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_drawdown(self, ax, equity_curve):
        """Plot drawdown chart"""
        
        equity = pd.Series(equity_curve)
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        
        time_index = pd.date_range(
            end=datetime.now(),
            periods=len(equity_curve),
            freq='15min'
        )
        
        # Plot drawdown
        ax.fill_between(time_index, drawdown, 0, 
                       where=(drawdown < 0), 
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(time_index, drawdown, 'r-', linewidth=1)
        
        # Mark maximum drawdown
        max_dd_idx = drawdown.idxmin()
        ax.plot(time_index[max_dd_idx], drawdown[max_dd_idx], 'ro', markersize=8)
        
        # Format
        ax.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add max drawdown text
        ax.text(0.02, 0.02, f'Max Drawdown: {drawdown.min():.2f}%',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    def _plot_returns_distribution(self, ax, trades):
        """Plot returns distribution histogram"""
        
        if not trades:
            ax.text(0.5, 0.5, 'No trades to display', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        returns = [t.get('return', 0) * 100 for t in trades]
        
        # Create histogram
        n, bins, patches = ax.hist(returns, bins=30, alpha=0.7, edgecolor='black')
        
        # Color bars by profit/loss
        for i, patch in enumerate(patches):
            if bins[i] >= 0:
                patch.set_facecolor(self.colors['success'])
            else:
                patch.set_facecolor(self.colors['danger'])
        
        # Add normal distribution overlay
        mu, sigma = np.mean(returns), np.std(returns)
        x = np.linspace(min(returns), max(returns), 100)
        ax.plot(x, n.max() * np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi)),
                'k--', linewidth=2, label='Normal Distribution')
        
        # Add vertical lines for mean and median
        ax.axvline(x=np.mean(returns), color='blue', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
        ax.axvline(x=np.median(returns), color='green', linestyle='--', 
                  linewidth=2, label=f'Median: {np.median(returns):.2f}%')
        
        # Format
        ax.set_title('Trade Returns Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_monthly_returns_heatmap(self, ax, equity_curve):
        """Plot monthly returns heatmap"""
        
        # Convert to daily returns first
        daily_equity = pd.Series(equity_curve).resample('D').last().dropna()
        
        if len(daily_equity) < 30:
            ax.text(0.5, 0.5, 'Insufficient data for monthly analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate monthly returns
        monthly_returns = daily_equity.pct_change().resample('M').apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        pivot_table = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        # Create heatmap data
        heatmap_data = pivot_table.pivot(index='Year', columns='Month', values='Return')
        
        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'Return (%)'}, ax=ax)
        
        # Format
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
    
    def _plot_rolling_sharpe(self, ax, equity_curve):
        """Plot rolling Sharpe ratio"""
        
        window = 252  # Approx 1 year of 15-min bars
        
        if len(equity_curve) < window:
            ax.text(0.5, 0.5, 'Insufficient data for rolling Sharpe', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        # Calculate rolling Sharpe
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252 * 96)
        
        time_index = pd.date_range(
            end=datetime.now(),
            periods=len(rolling_sharpe),
            freq='15min'
        )
        
        # Plot
        ax.plot(time_index, rolling_sharpe, 'b-', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        ax.axhline(y=2, color='gold', linestyle='--', alpha=0.5, label='Sharpe = 2')
        
        # Fill areas
        ax.fill_between(time_index, 0, rolling_sharpe, 
                       where=(rolling_sharpe > 0), 
                       color='green', alpha=0.2)
        ax.fill_between(time_index, 0, rolling_sharpe, 
                       where=(rolling_sharpe < 0), 
                       color='red', alpha=0.2)
        
        # Format
        ax.set_title(f'Rolling Sharpe Ratio ({window} periods)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_win_loss_analysis(self, ax, trades):
        """Plot win/loss analysis"""
        
        if not trades:
            ax.text(0.5, 0.5, 'No trades to analyze', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Categorize trades
        wins = [t['return'] * 100 for t in trades if t.get('return', 0) > 0]
        losses = [abs(t['return'] * 100) for t in trades if t.get('return', 0) <= 0]
        
        # Create box plot data
        data_to_plot = []
        labels = []
        
        if wins:
            data_to_plot.append(wins)
            labels.append(f'Wins (n={len(wins)})')
        
        if losses:
            data_to_plot.append(losses)
            labels.append(f'Losses (n={len(losses)})')
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors_list = [self.colors['success'], self.colors['danger']]
        for patch, color in zip(bp['boxes'], colors_list[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add mean markers
        for i, data in enumerate(data_to_plot):
            ax.scatter(i+1, np.mean(data), color='black', s=100, zorder=3, marker='D')
        
        # Format
        ax.set_title('Win/Loss Distribution Analysis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        if wins and losses:
            avg_win = np.mean(wins)
            avg_loss = np.mean(losses)
            win_rate = len(wins) / (len(wins) + len(losses)) * 100
            
            stats_text = f'Win Rate: {win_rate:.1f}%\n'
            stats_text += f'Avg Win: {avg_win:.2f}%\n'
            stats_text += f'Avg Loss: {avg_loss:.2f}%\n'
            stats_text += f'Risk/Reward: {avg_win/avg_loss:.2f}'
            
            ax.text(0.98, 0.98, stats_text,
                   transform=ax.transAxes, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_trade_duration(self, ax, trades):
        """Plot trade duration analysis"""
        
        if not trades or not any('duration' in t for t in trades):
            ax.text(0.5, 0.5, 'No duration data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get durations for wins and losses
        win_durations = [t.get('duration', 0) for t in trades if t.get('return', 0) > 0]
        loss_durations = [t.get('duration', 0) for t in trades if t.get('return', 0) <= 0]
        
        # Create histogram
        if win_durations:
            ax.hist(win_durations, bins=20, alpha=0.6, 
                   label=f'Winning Trades', color=self.colors['success'])
        
        if loss_durations:
            ax.hist(loss_durations, bins=20, alpha=0.6, 
                   label=f'Losing Trades', color=self.colors['danger'])
        
        # Add vertical lines for averages
        if win_durations:
            ax.axvline(x=np.mean(win_durations), color=self.colors['success'], 
                      linestyle='--', linewidth=2)
        
        if loss_durations:
            ax.axvline(x=np.mean(loss_durations), color=self.colors['danger'], 
                      linestyle='--', linewidth=2)
        
        # Format
        ax.set_title('Trade Duration Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Duration (hours)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_strategy_comparison(self, ax, trades):
        """Plot cumulative returns by strategy"""
        
        # Group trades by strategy
        strategies = {}
        for trade in trades:
            strategy = trade.get('strategy', 'Unknown')
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(trade)
        
        if not strategies:
            ax.text(0.5, 0.5, 'No strategy data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        colors = plt.cm.Set3(range(len(strategies)))
        
        for i, (strategy_name, strategy_trades) in enumerate(strategies.items()):
            if not strategy_trades:
                continue
                
            # Calculate cumulative returns
            cumulative_returns = [1.0]
            for trade in sorted(strategy_trades, key=lambda x: x.get('entry_time', datetime.now())):
                cumulative_returns.append(
                    cumulative_returns[-1] * (1 + trade.get('return', 0))
                )
            
            # Create time index
            time_index = pd.date_range(
                start=datetime.now() - timedelta(days=30),
                periods=len(cumulative_returns),
                freq='D'
            )[:len(cumulative_returns)]
            
            ax.plot(time_index, cumulative_returns, 
                   label=strategy_name, color=colors[i], linewidth=2)
        
        ax.set_title('Cumulative Returns by Strategy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _generate_html_report(self, metrics: Dict, trades: List[Dict], 
                            config: Dict = None) -> str:
        """Generate HTML report"""
        
        html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>Trading Performance Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
        }
        .metric-card {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            margin: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: inline-block;
            min-width: 200px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            font-size: 14px;
            color: #7f8c8d;
        }
        .positive {
            color: #27ae60;
        }
        .negative {
            color: #e74c3c;
        }
        .section {
            background-color: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #34495e;
            color: white;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Algorithmic Trading Performance Report</h1>
        <p>Generated on {timestamp}</p>
    </div>
    
    <div class="section">
        <h2>Key Performance Metrics</h2>
        <div style="text-align: center;">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {return_class}">{total_return:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{sharpe_ratio:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{max_drawdown:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{win_rate:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{profit_factor:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{total_trades}</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Performance Charts</h2>
        <div class="chart-container">
            <img src="performance_charts.png" alt="Performance Charts">
        </div>
    </div>
    
    <div class="section">
        <h2>Detailed Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>CAGR</td>
                <td>{cagr:.2f}%</td>
                <td>Compound Annual Growth Rate</td>
            </tr>
            <tr>
                <td>Volatility</td>
                <td>{volatility:.2f}%</td>
                <td>Annualized standard deviation</td>
            </tr>
            <tr>
                <td>Sortino Ratio</td>
                <td>{sortino_ratio:.3f}</td>
                <td>Risk-adjusted return (downside only)</td>
            </tr>
            <tr>
                <td>Calmar Ratio</td>
                <td>{calmar_ratio:.3f}</td>
                <td>CAGR / Max Drawdown</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td>{avg_win:.2f}%</td>
                <td>Average winning trade return</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td>{avg_loss:.2f}%</td>
                <td>Average losing trade return</td>
            </tr>
            <tr>
                <td>Risk/Reward Ratio</td>
                <td>{risk_reward_ratio:.2f}</td>
                <td>Average win / Average loss</td>
            </tr>
            <tr>
                <td>Expectancy</td>
                <td>{expectancy:.2f}%</td>
                <td>Expected return per trade</td>
            </tr>
            <tr>
                <td>Kelly %</td>
                <td>{kelly_percentage:.1f}%</td>
                <td>Optimal position size</td>
            </tr>
            <tr>
                <td>Max Consecutive Wins</td>
                <td>{max_consecutive_wins}</td>
                <td>Longest winning streak</td>
            </tr>
            <tr>
                <td>Max Consecutive Losses</td>
                <td>{max_consecutive_losses}</td>
                <td>Longest losing streak</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Configuration</h2>
        <pre>{config_json}</pre>
    </div>
</body>
</html>
'''
        
        # Format values
        return_class = 'positive' if metrics['total_return'] > 0 else 'negative'
        config_json = json.dumps(config, indent=2) if config else '{}'
        
        # Fill template
        html = html_template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            return_class=return_class,
            config_json=config_json,
            **metrics
        )
        
        return html
    
    def _safe_calc(self, func):
        """Safely calculate a metric"""
        try:
            return func()
        except:
            return 0
    
    def _calculate_cagr(self, equity_curve):
        """Calculate CAGR"""
        years = len(equity_curve) / (252 * 96)  # 15-min bars
        if years <= 0:
            return 0
        return ((equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1) * 100
    
    def _calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252 * 96)
    
    def _calculate_sortino_ratio(self, returns):
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        return (returns.mean() / downside_returns.std()) * np.sqrt(252 * 96)
    
    def _calculate_calmar_ratio(self, equity_curve):
        """Calculate Calmar ratio"""
        cagr = self._calculate_cagr(equity_curve)
        max_dd = abs(self._calculate_max_drawdown(equity_curve))
        if max_dd == 0:
            return 0
        return cagr / max_dd
    
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return drawdown.min() * 100
    
    def _calculate_max_dd_duration(self, equity_curve):
        """Calculate maximum drawdown duration"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        
        # Calculate durations
        max_duration = 0
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
                
        return max_duration
    
    def _calculate_profit_factor(self, trades):
        """Calculate profit factor"""
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        return gross_profit / gross_loss
    
    def _calculate_kelly_percentage(self, trades):
        """Calculate Kelly percentage"""
        if not trades:
            return 0
            
        wins = [t['return'] for t in trades if t.get('return', 0) > 0]
        losses = [abs(t['return']) for t in trades if t.get('return', 0) < 0]
        
        if not wins or not losses:
            return 0
            
        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 0
            
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return max(0, min(kelly * 100, 25))
    
    def _calculate_ulcer_index(self, equity_curve):
        """Calculate Ulcer Index"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak * 100
        return np.sqrt(np.mean(drawdown ** 2))
    
    def _calculate_recovery_factor(self, equity_curve):
        """Calculate recovery factor"""
        total_return = equity_curve[-1] - equity_curve[0]
        max_dd = abs(self._calculate_max_drawdown(equity_curve) / 100 * equity_curve[0])
        
        if max_dd == 0:
            return 0
        return total_return / max_dd
    
    def _calculate_payoff_ratio(self, trades):
        """Calculate payoff ratio"""
        wins = [t['return'] for t in trades if t.get('return', 0) > 0]
        losses = [abs(t['return']) for t in trades if t.get('return', 0) < 0]
        
        if not wins or not losses:
            return 0
            
        return np.mean(wins) / np.mean(losses)
    
    def _calculate_max_consecutive(self, returns, wins=True):
        """Calculate max consecutive wins/losses"""
        max_count = 0
        current_count = 0
        
        for r in returns:
            if (wins and r > 0) or (not wins and r <= 0):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
                
        return max_count
    
    def _calculate_monthly_returns(self, returns):
        """Calculate average monthly returns"""
        # Simplified - assumes 21 trading days per month
        periods_per_month = 21 * 96  # 96 periods per day for 15-min bars
        
        if len(returns) < periods_per_month:
            return np.mean(returns) * periods_per_month * 100
        
        # Group returns by month
        monthly_returns = []
        for i in range(0, len(returns), periods_per_month):
            month_returns = returns[i:i+periods_per_month]
            if len(month_returns) > 0:
                monthly_return = (1 + month_returns).prod() - 1
                monthly_returns.append(monthly_return)
        
        return np.mean(monthly_returns) * 100 if monthly_returns else 0