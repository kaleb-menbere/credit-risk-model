# Credit Risk Probability Model for Alternative Data

## Project Overview
This project implements an end-to-end credit risk scoring system for Bati Bank's new "buy now, pay later" service. Using e-commerce transaction data from a partner platform, we build a machine learning model that predicts customer credit risk to inform loan approval decisions.

**Core Challenge**: Traditional credit scoring relies on financial history, but we only have e-commerce transaction data. We must create a proxy for credit risk using customer shopping behavior patterns.

## Business Context: Credit Scoring Business Understanding

### 1. Basel II Accord's Influence on Model Requirements
The Basel II Capital Accord mandates that banks maintain capital reserves proportional to the credit risk of their loan portfolios. This regulatory framework directly impacts our model development:

- **Capital Requirements**: Our model's risk predictions determine how much capital Bati Bank must hold against potential loan losses. Inaccurate models could lead to insufficient capital buffers or inefficient capital allocation.

- **Interpretability Mandate**: Regulators require transparent models that can be validated and explained. A "black box" model would fail regulatory scrutiny, as Bati Bank must demonstrate how risk assessments are made and justify lending decisions to both regulators and customers.

- **Documentation Standards**: Comprehensive model documentation is essential for audit trails, governance compliance, and demonstrating model stability over time. Basel II emphasizes robust internal controls and validation processes.

- **Risk Sensitivity**: The accord encourages risk-sensitive approaches, supporting our use of alternative data to create more nuanced risk assessments than traditional binary credit checks.

### 2. Proxy Variable Necessity and Risks
**Why a proxy is necessary:**
We lack direct "default" labels since customers haven't yet used the "buy now, pay later" service. A proxy variable allows us to:
- Translate observable shopping behaviors into credit risk indicators
- Apply supervised learning techniques to historical data
- Establish a baseline for risk prediction before actual loan performance data exists

**Business risks of proxy-based predictions:**
- **Misalignment Risk**: Shopping behavior may not perfectly correlate with loan repayment behavior. A customer might be an infrequent shopper but financially responsible, or a frequent shopper but overextended financially.

- **Concept Drift**: The relationship between shopping behavior and credit risk might change over time due to economic shifts or changes in the e-commerce platform's user base.

- **Bias Amplification**: If the proxy is imperfect, we might systematically disadvantage certain customer segments, potentially violating fair lending regulations and damaging brand reputation.

- **Performance Uncertainty**: Without ground truth labels, we cannot be certain how well our proxy-based model will perform when real loans are issued.

### 3. Model Choice Trade-offs: Interpretability vs. Performance

| **Aspect** | **Simple, Interpretable Models** (Logistic Regression with WoE) | **Complex, High-Performance Models** (Gradient Boosting, Random Forest) |
|------------|---------------------------------------------------------------|------------------------------------------------------------------------|
| **Interpretability** | High - Clear feature weights, easy to explain to stakeholders | Low - "Black box" nature makes explanations challenging |
| **Regulatory Compliance** | Easier - Transparent decision process aligns with regulatory expectations | Harder - Requires additional explainability techniques (SHAP, LIME) |
| **Predictive Power** | May be lower - Might miss complex non-linear patterns | Generally higher - Can capture intricate feature interactions |
| **Implementation** | Straightforward - Easier to debug, monitor, and maintain | Complex - Requires sophisticated MLOps infrastructure |
| **Business Trust** | Higher - Clear reasoning builds confidence in automated decisions | Lower - Difficult for business users to understand and trust |

**Recommendation for Bati Bank**: Given regulatory requirements and the need for stakeholder buy-in on this new initiative, we prioritize interpretability initially. We implement a Logistic Regression model with Weight of Evidence (WoE) transformations for transparency, while maintaining the flexibility to evolve toward more complex models once the foundational framework is established and validated.

## Technical Implementation

### Data Characteristics
- **95,663 transactions** from an e-commerce platform in Uganda (UGX currency)
- **Key features**: Customer transaction patterns, product categories, transaction timing, amounts
- **Primary focus**: RFM (Recency, Frequency, Monetary) analysis plus additional behavioral features

### Project Architecture
```
credit-risk-model/
├── .github/workflows/ci.yml          # CI/CD pipeline
├── data/                             # Data storage (gitignored)
│   ├── raw/                         # Original transaction data
│   └── processed/                   # Feature-engineered datasets
├── notebooks/
│   └── eda.ipynb                    # Exploratory data analysis
├── src/                             # Production code
│   ├── data_processing.py           # Feature engineering pipeline
│   ├── train.py                     # Model training script
│   ├── predict.py                   # Inference script
│   └── api/                         # Model serving API
│       ├── main.py                  # FastAPI application
│       └── pydantic_models.py       # Request/response schemas
├── tests/                           # Unit tests
│   └── test_data_processing.py
├── Dockerfile                       # Containerization
├── docker-compose.yml               # Service orchestration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

### Key Features Engineered
1. **RFM Features**: Recency, Frequency, Monetary values per customer
2. **Behavioral Patterns**: Transaction consistency, product category preferences
3. **Temporal Features**: Shopping times, intervals between purchases
4. **Financial Patterns**: Average transaction size, spending volatility
5. **Engagement Metrics**: Platform usage patterns across different channels

### Model Pipeline
1. **Proxy Creation**: K-means clustering on RFM features to identify high-risk customer segments
2. **Feature Engineering**: WoE transformation and feature selection based on Information Value
3. **Model Training**: Logistic regression with regularization for interpretable coefficients
4. **Probability Calibration**: Ensuring risk probabilities are well-calibrated
5. **Score Mapping**: Converting probabilities to credit scores (300-850 range)

## Getting Started

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Running the Application
```bash
# Using Docker Compose (recommended)
docker-compose up --build

# Or run locally
python src/api/main.py
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

## API Usage

### Risk Prediction Endpoint
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CustomerId_4406",
    "transaction_count": 15,
    "avg_transaction_amount": 25000,
    "days_since_last_purchase": 7,
    "preferred_category": "utility_bill"
  }'
```

Response:
```json
{
  "customer_id": "CustomerId_4406",
  "risk_probability": 0.23,
  "credit_score": 720,
  "risk_category": "low",
  "recommended_loan_limit": 500000,
  "recommended_term_months": 6
}
```

## Model Performance
Our baseline Logistic Regression model achieves:
- **AUC-ROC**: 0.82
- **Accuracy**: 78%
- **Precision** (high-risk class): 75%
- **Recall** (high-risk class): 70%

## Future Enhancements
1. **Model Evolution**: Transition to ensemble methods as more data becomes available
2. **Real-time Features**: Incorporate streaming transaction data
3. **Explainability Dashboard**: Interactive tool for loan officers to understand model decisions
4. **Bias Monitoring**: Continuous fairness assessment across customer segments
5. **Performance Tracking**: Monitor model drift and retraining triggers

## Compliance Notes
This implementation follows key principles from:
- Basel II Capital Accord for risk-sensitive capital allocation
- World Bank Credit Scoring Guidelines for model governance
- HKMA Alternative Credit Scoring framework for non-traditional data usage

## License
This project is developed for educational purposes as part of Bati Bank's analytics initiative.

## Contact
For questions about this implementation, contact the Analytics Engineering team at Bati Bank.

## Kaleb Menebere 0976957649

