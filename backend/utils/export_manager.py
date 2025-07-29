#!/usr/bin/env python3
"""
Export Manager for Candlestick Pattern Prediction System
Handles CSV and PDF export functionality for predictions and reports
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import os
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.colors import HexColor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns

class ExportManager:
    """Manages export functionality for predictions and reports"""
    
    def __init__(self, export_dir: str = None):
        self.logger = logging.getLogger(__name__)
        self.export_dir = export_dir or os.path.join(os.getcwd(), 'exports')
        os.makedirs(self.export_dir, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def export_predictions_csv(self, 
                             predictions: List[Dict], 
                             user_info: Dict, 
                             symbol: str, 
                             timeframe: str,
                             model_type: str) -> str:
        """Export predictions to CSV format"""
        try:
            # Create DataFrame from predictions
            df = pd.DataFrame(predictions)
            
            # Add metadata columns
            df['user_id'] = user_info.get('id', 'unknown')
            df['username'] = user_info.get('username', 'unknown')
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            df['model_type'] = model_type
            df['export_timestamp'] = datetime.now().isoformat()
            
            # Reorder columns for better readability
            column_order = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'predicted', 'confidence', 'bullish', 'symbol', 'timeframe',
                'model_type', 'user_id', 'username', 'export_timestamp'
            ]
            
            # Only include columns that exist
            available_columns = [col for col in column_order if col in df.columns]
            df = df[available_columns]
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"predictions_{symbol}_{timeframe}_{model_type}_{timestamp}.csv"
            filepath = os.path.join(self.export_dir, filename)
            
            # Export to CSV
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Predictions exported to CSV: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting predictions to CSV: {e}")
            raise
    
    def export_model_performance_csv(self, 
                                   performance_data: List[Dict], 
                                   user_info: Dict) -> str:
        """Export model performance metrics to CSV"""
        try:
            # Create DataFrame from performance data
            df = pd.DataFrame(performance_data)
            
            # Add metadata
            df['user_id'] = user_info.get('id', 'unknown')
            df['username'] = user_info.get('username', 'unknown')
            df['export_timestamp'] = datetime.now().isoformat()
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"model_performance_{user_info.get('username', 'user')}_{timestamp}.csv"
            filepath = os.path.join(self.export_dir, filename)
            
            # Export to CSV
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Model performance exported to CSV: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting model performance to CSV: {e}")
            raise
    
    def create_prediction_chart(self, 
                              historical_data: pd.DataFrame, 
                              predictions: List[Dict],
                              symbol: str,
                              timeframe: str) -> str:
        """Create a chart for predictions"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot historical candlestick data (simplified as line chart)
            if not historical_data.empty:
                ax1.plot(historical_data.index, historical_data['close'], 
                        label='Historical Close', color='blue', linewidth=1.5)
            
            # Plot predictions
            if predictions:
                pred_df = pd.DataFrame(predictions)
                if 'timestamp' in pred_df.columns:
                    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
                    ax1.plot(pred_df['timestamp'], pred_df['close'], 
                            label='Predicted Close', color='red', 
                            linewidth=2, linestyle='--', marker='o', markersize=4)
            
            ax1.set_title(f'{symbol} - {timeframe} Candlestick Predictions', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot volume if available
            if not historical_data.empty and 'volume' in historical_data.columns:
                ax2.bar(historical_data.index, historical_data['volume'], 
                       alpha=0.7, color='gray', label='Volume')
                ax2.set_ylabel('Volume', fontsize=12)
                ax2.set_xlabel('Time', fontsize=12)
                ax2.legend()
            
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_filename = f"chart_{symbol}_{timeframe}_{timestamp}.png"
            chart_path = os.path.join(self.export_dir, chart_filename)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Error creating prediction chart: {e}")
            return None
    
    def export_predictions_pdf(self, 
                             predictions: List[Dict], 
                             user_info: Dict, 
                             symbol: str, 
                             timeframe: str,
                             model_type: str,
                             historical_data: pd.DataFrame = None,
                             performance_metrics: Dict = None) -> str:
        """Export predictions to PDF format with charts and analysis"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"prediction_report_{symbol}_{timeframe}_{timestamp}.pdf"
            filepath = os.path.join(self.export_dir, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                textColor=colors.darkblue,
                alignment=1  # Center alignment
            )
            
            title = Paragraph(f"Candlestick Pattern Prediction Report", title_style)
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Report metadata
            metadata_style = styles['Normal']
            metadata = [
                f"<b>Symbol:</b> {symbol}",
                f"<b>Timeframe:</b> {timeframe}",
                f"<b>Model Type:</b> {model_type}",
                f"<b>User:</b> {user_info.get('username', 'Unknown')}",
                f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"<b>Total Predictions:</b> {len(predictions)}"
            ]
            
            for item in metadata:
                story.append(Paragraph(item, metadata_style))
                story.append(Spacer(1, 6))
            
            story.append(Spacer(1, 20))
            
            # Performance metrics if available
            if performance_metrics:
                story.append(Paragraph("<b>Model Performance Metrics</b>", styles['Heading2']))
                story.append(Spacer(1, 10))
                
                metrics_data = [['Metric', 'Value']]
                for key, value in performance_metrics.items():
                    if isinstance(value, float):
                        metrics_data.append([key.replace('_', ' ').title(), f"{value:.4f}"])
                    else:
                        metrics_data.append([key.replace('_', ' ').title(), str(value)])
                
                metrics_table = Table(metrics_data)
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(metrics_table)
                story.append(Spacer(1, 20))
            
            # Add chart if historical data is available
            if historical_data is not None:
                chart_path = self.create_prediction_chart(historical_data, predictions, symbol, timeframe)
                if chart_path and os.path.exists(chart_path):
                    story.append(Paragraph("<b>Price Prediction Chart</b>", styles['Heading2']))
                    story.append(Spacer(1, 10))
                    
                    # Add chart image
                    img = Image(chart_path, width=6*inch, height=5*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
            
            # Predictions table
            story.append(Paragraph("<b>Detailed Predictions</b>", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            # Create predictions table
            if predictions:
                # Limit to first 20 predictions for PDF readability
                limited_predictions = predictions[:20]
                
                table_data = [['Timestamp', 'Open', 'High', 'Low', 'Close', 'Predicted', 'Confidence']]
                
                for pred in limited_predictions:
                    row = [
                        pred.get('timestamp', 'N/A')[:16] if pred.get('timestamp') else 'N/A',
                        f"{pred.get('open', 0):.2f}",
                        f"{pred.get('high', 0):.2f}",
                        f"{pred.get('low', 0):.2f}",
                        f"{pred.get('close', 0):.2f}",
                        'Yes' if pred.get('predicted', False) else 'No',
                        f"{pred.get('confidence', 0):.2f}" if pred.get('confidence') else 'N/A'
                    ]
                    table_data.append(row)
                
                predictions_table = Table(table_data)
                predictions_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(predictions_table)
                
                if len(predictions) > 20:
                    story.append(Spacer(1, 10))
                    story.append(Paragraph(f"<i>Note: Showing first 20 of {len(predictions)} predictions. "
                                          f"Full data available in CSV export.</i>", styles['Italic']))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"Predictions exported to PDF: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting predictions to PDF: {e}")
            raise
    
    def cleanup_old_exports(self, days_old: int = 7):
        """Clean up export files older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            for filename in os.listdir(self.export_dir):
                filepath = os.path.join(self.export_dir, filename)
                if os.path.isfile(filepath):
                    file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_modified < cutoff_date:
                        os.remove(filepath)
                        self.logger.info(f"Cleaned up old export file: {filename}")
                        
        except Exception as e:
            self.logger.error(f"Error cleaning up old exports: {e}")
    
    def export_model_performance_pdf(self,
                                  performance_data: List[Dict],
                                  user_info: Dict) -> str:
        """Export model performance metrics to PDF format with charts and analysis"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"model_performance_report_{user_info.get('username', 'user')}_{timestamp}.pdf"
            filepath = os.path.join(self.export_dir, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                textColor=colors.darkblue,
                alignment=1  # Center alignment
            )
            
            title = Paragraph(f"Model Performance Report", title_style)
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Report metadata
            metadata_style = styles['Normal']
            metadata = [
                f"<b>User:</b> {user_info.get('username', 'Unknown')}",
                f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"<b>Total Models:</b> {len(performance_data)}"
            ]
            
            for item in metadata:
                story.append(Paragraph(item, metadata_style))
                story.append(Spacer(1, 6))
            
            story.append(Spacer(1, 20))
            
            # Create performance comparison chart
            if performance_data:
                story.append(Paragraph("<b>Model Performance Comparison</b>", styles['Heading2']))
                story.append(Spacer(1, 10))
                
                # Create a bar chart for model comparison
                plt.figure(figsize=(10, 6))
                
                # Extract model names and accuracy values
                model_names = [p.get('model_type', f"Model {i}") for i, p in enumerate(performance_data)]
                accuracy_values = [p.get('accuracy', 0) for p in performance_data]
                
                # Create bar chart
                plt.bar(model_names, accuracy_values, color='skyblue')
                plt.xlabel('Model Type')
                plt.ylabel('Accuracy')
                plt.title('Model Accuracy Comparison')
                plt.ylim(0, 1.0)  # Assuming accuracy is between 0 and 1
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Save chart
                chart_filename = f"performance_chart_{timestamp}.png"
                chart_path = os.path.join(self.export_dir, chart_filename)
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Add chart to PDF
                img = Image(chart_path, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 20))
            
            # Detailed performance metrics table
            story.append(Paragraph("<b>Detailed Model Performance</b>", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            if performance_data:
                # Define table headers
                headers = ['Model Type', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time']
                
                # Create table data
                table_data = [headers]
                
                for model in performance_data:
                    row = [
                        model.get('model_type', 'Unknown'),
                        f"{model.get('accuracy', 0):.4f}",
                        f"{model.get('precision', 0):.4f}",
                        f"{model.get('recall', 0):.4f}",
                        f"{model.get('f1_score', 0):.4f}",
                        f"{model.get('training_time', 0):.2f}s"
                    ]
                    table_data.append(row)
                
                # Create table
                performance_table = Table(table_data)
                performance_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(performance_table)
            else:
                story.append(Paragraph("No performance data available", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"Model performance exported to PDF: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting model performance to PDF: {e}")
            raise
    
    def export_historical_data_csv(self,
                                historical_data: pd.DataFrame,
                                user_info: Dict,
                                symbol: str,
                                timeframe: str,
                                period: str) -> str:
        """Export historical candlestick data to CSV format"""
        try:
            # Create a copy of the DataFrame to avoid modifying the original
            df = historical_data.copy()
            
            # Reset index to make timestamp a column
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.rename(columns={'index': 'timestamp'}, inplace=True)
            
            # Add metadata columns
            df['user_id'] = user_info.get('id', 'unknown')
            df['username'] = user_info.get('username', 'unknown')
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            df['period'] = period
            df['export_timestamp'] = datetime.now().isoformat()
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"historical_data_{symbol}_{timeframe}_{period}_{timestamp}.csv"
            filepath = os.path.join(self.export_dir, filename)
            
            # Export to CSV
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Historical data exported to CSV: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting historical data to CSV: {e}")
            raise
    
    def get_export_stats(self) -> Dict:
        """Get statistics about export files"""
        try:
            stats = {
                'total_files': 0,
                'total_size_mb': 0,
                'csv_files': 0,
                'pdf_files': 0,
                'chart_files': 0,
                'oldest_file': None,
                'newest_file': None
            }
            
            if not os.path.exists(self.export_dir):
                return stats
            
            oldest_time = None
            newest_time = None
            
            for filename in os.listdir(self.export_dir):
                filepath = os.path.join(self.export_dir, filename)
                if os.path.isfile(filepath):
                    stats['total_files'] += 1
                    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                    stats['total_size_mb'] += file_size
                    
                    # Count by type
                    if filename.endswith('.csv'):
                        stats['csv_files'] += 1
                    elif filename.endswith('.pdf'):
                        stats['pdf_files'] += 1
                    elif filename.endswith('.png'):
                        stats['chart_files'] += 1
                    
                    # Track oldest and newest
                    file_time = os.path.getmtime(filepath)
                    if oldest_time is None or file_time < oldest_time:
                        oldest_time = file_time
                        stats['oldest_file'] = filename
                    if newest_time is None or file_time > newest_time:
                        newest_time = file_time
                        stats['newest_file'] = filename
            
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting export stats: {e}")
            return {}