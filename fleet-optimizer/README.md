# Fleet Optimizer - AI-Powered Real-Time Fleet Management

## 🚀 Overview
An intelligent fleet optimization system that uses AI, real-time traffic data, and weather information to optimize vehicle allocation and routing decisions.

## ✨ Features

### 🤖 AI-Powered Optimization
- **Groq AI Integration**: Uses llama-3.1-8b-instant for intelligent decision making
- **Real-time Context**: AI considers traffic, weather, and demand patterns
- **Smart Allocation**: Optimizes vehicle assignments based on multiple factors

### 🌐 Real-Time Data Integration
- **Google Maps API**: Live traffic data and route optimization
- **OpenWeather API**: Weather conditions affecting travel times
- **Dynamic Updates**: Real-time data refresh every 5 seconds

### 📊 Advanced Analytics
- **Vehicle Utilization**: Track efficiency and performance metrics
- **Customer Satisfaction**: AI-calculated satisfaction scores
- **Financial Performance**: Revenue tracking and cost analysis
- **Comprehensive Reporting**: Detailed performance insights

### 🎯 Key Capabilities
- **Auto-Allocation**: Automatic vehicle assignment every 5 seconds
- **Priority-Based Routing**: Handles different demand priorities (1-5)
- **Multi-Modal Support**: Rides, deliveries, and emergency services
- **Interactive Dashboard**: Real-time visualization with Gradio
- **AI Assistant**: Groq-powered chatbot for system insights

## 🛠️ Installation

### Prerequisites
```bash
pip install gradio plotly requests python-dotenv
```

### Environment Setup
Create a `.env` file with your API keys:
```bash
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
GROQ_API_KEY=your_groq_api_key
```

## 🚀 Usage

### Quick Start
```bash
python fleet_optimizer.py
```

### Features Available
1. **Start Auto Allocation**: Begin automatic vehicle processing
2. **Real-time Dashboard**: Live map with vehicle and demand visualization
3. **AI Assistant**: Ask questions about allocations and system decisions
4. **Comprehensive Reports**: Generate detailed performance analytics
5. **API Testing**: Verify all external API connections

## 📈 Performance Metrics

### Real-Time Monitoring
- Vehicle utilization rates
- Demand fulfillment statistics
- Customer satisfaction scores
- API response times
- System optimization levels

### AI Recommendations
- Traffic-aware routing
- Weather-optimized assignments
- Priority-based allocation
- Capacity utilization optimization

## 🔧 Technical Architecture

### Core Components
- **FleetOptimizer**: Main optimization engine
- **RealtimeAPIClient**: External API integration
- **Vehicle Management**: Dynamic vehicle tracking
- **Demand Processing**: Intelligent demand handling
- **AI Integration**: Groq-powered decision making

### Data Flow
1. **Real-time Data Collection**: Traffic, weather, and demand data
2. **AI Processing**: Groq AI analyzes conditions and provides recommendations
3. **Optimization Engine**: Multi-factor vehicle allocation algorithm
4. **Dashboard Updates**: Real-time visualization and reporting

## 🎯 Use Cases

### Fleet Management
- Ride-sharing optimization
- Delivery route planning
- Emergency vehicle dispatch
- Multi-modal transportation

### Business Intelligence
- Performance analytics
- Cost optimization
- Customer satisfaction tracking
- Operational efficiency metrics

## 🔒 Security
- Environment variable-based API key management
- No hardcoded secrets in source code
- Secure API communication
- GitHub secret scanning compliant

## 📊 Sample Output

```
🤖 AI OPTIMIZATION RECOMMENDATIONS (via Groq API):

📊 Current Status:
• Available Vehicles: 15
• Pending Demands: 8
• Traffic Conditions: medium
• Weather: rain

🎯 AI Recommendations:
Based on current conditions, prioritize high-priority demands and 
consider weather impact on travel times. Reallocate vehicles from 
low-demand areas to high-demand zones.

📈 Real-time Data Integration:
• Traffic API: ✅ Active
• Weather API: ✅ Active  
• AI Processing: ✅ Active
```

## 🤝 Contributing
This project is part of the DataSprint initiative. Contributions are welcome!

## 📄 License
Part of the DataSprint project repository.
