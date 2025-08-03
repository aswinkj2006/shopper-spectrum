# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations

A comprehensive e-commerce analytics project that performs customer segmentation using RFM analysis and provides product recommendations using collaborative filtering.

## 📁 Project Structure

```
shopper-spectrum/
├── shopper_spectrum_analysis.ipynb    # Main analysis notebook
├── streamlit_app.py                   # Streamlit web application
├── requirements.txt                   # Python dependencies
├── online_retail.csv                  # Dataset (you need to add this)
├── README.md                         # This file
└── models/                           # Generated model files
    ├── customer_segmentation_model.pkl
    ├── rfm_scaler.pkl
    ├── cluster_labels.pkl
    ├── product_similarity_matrix.pkl
    ├── product_mapping.pkl
    └── rfm_sample_data.csv
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or create project directory
mkdir shopper-spectrum
cd shopper-spectrum

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the dataset from the provided link and save it as `online_retail.csv` in the project directory.

Dataset link: [Online Retail Dataset](https://drive.google.com/file/d/1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-/view?usp=sharing)

### 3. Run Analysis Notebook

```bash
# Start Jupyter notebook
jupyter notebook

# Open and run shopper_spectrum_analysis.ipynb
# This will generate all the required model files
```

### 4. Launch Streamlit App

```bash
# Run the Streamlit application
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Features

### 🎯 Product Recommendations
- Search products by name or select from dropdown
- Get 5 similar products using collaborative filtering
- Similarity scores based on customer purchase patterns

### 👥 Customer Segmentation
- Input customer RFM metrics (Recency, Frequency, Monetary)
- Predict customer segment: High-Value, Regular, Occasional, or At-Risk
- Get targeted marketing strategies for each segment

### 📈 Analytics Dashboard
- Customer segment distribution visualization
- RFM metrics analysis
- Interactive scatter plots
- Key performance indicators

## 🔧 Technical Implementation

### Machine Learning Models
- **K-Means Clustering**: For customer segmentation
- **Collaborative Filtering**: For product recommendations
- **Cosine Similarity**: For product similarity calculation

### Key Libraries
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web App**: Streamlit

## 📋 Dataset Description

| Column | Description |
|--------|-------------|
| InvoiceNo | Transaction number |
| StockCode | Unique product/item code |
| Description | Name of the product |
| Quantity | Number of products purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per product |
| CustomerID | Unique identifier for each customer |
| Country | Country where the customer is based |

## 🎯 Business Impact

### Customer Segments
1. **High-Value (🌟)**: Recent, frequent, high-spending customers
2. **Regular (👤)**: Steady purchasers with moderate patterns  
3. **Occasional (🔄)**: Infrequent buyers with growth potential
4. **At-Risk (⚠️)**: Haven't purchased recently, need retention

### Use Cases
- Targeted marketing campaigns
- Personalized product recommendations
- Customer retention strategies
- Inventory optimization
- Dynamic pricing strategies

## 🛠️ Model Performance

The project includes evaluation metrics:
- Silhouette Score for clustering quality
- Elbow method for optimal cluster selection
- Product similarity matrix for recommendations

## 📝 Usage Examples

### Product Recommendations
```python
# Example: Get recommendations for a product
product_code = "84879"  # Example product code
recommendations = get_product_recommendations(product_code, n_recommendations=5)
```

### Customer Segmentation
```python
# Example: Predict customer segment
customer_metrics = [30, 5, 500]  # [Recency, Frequency, Monetary]
segment = predict_customer_segment(customer_metrics)
```

## 🔄 Model Updates

To retrain models with new data:
1. Replace `online_retail.csv` with updated dataset
2. Run the complete Jupyter notebook
3. New model files will be generated automatically
4. Restart the Streamlit app to use updated models

## 📊 Performance Optimization

- Models are cached using `@st.cache_data` for faster loading
- Sample data used for dashboard to improve performance
- Efficient similarity calculations using vectorized operations

## 🎨 UI Features

- Responsive design with custom CSS
- Interactive visualizations with Plotly
- Color-coded customer segments
- Real-time predictions
- Professional styling and animations

## 🔍 Troubleshooting

### Common Issues:

1. **FileNotFoundError**: Make sure to run the Jupyter notebook first to generate model files
2. **Memory Issues**: Reduce sample size in the notebook if working with large datasets
3. **Slow Loading**: Check if all model files are properly cached

### Performance Tips:
- Use virtual environment for consistent dependencies
- Ensure sufficient RAM for large datasets
- Cache model files for faster subsequent loads

## 📈 Future Enhancements

- Deep learning-based recommendations
- Real-time data pipeline integration
- A/B testing framework
- Advanced customer lifetime value prediction
- Multi-language support
- Mobile app version

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the Jupyter notebook outputs
3. Verify all dependencies are installed correctly

---

**Built with ❤️ by Aswin**

