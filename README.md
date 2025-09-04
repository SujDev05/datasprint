# 🚖 Fleet Resource Optimization with AI Agents

A comprehensive fleet management system that uses AI agents to optimize vehicle allocation based on real-time traffic, weather data, and demand patterns.

## ✨ Features

- **Dataset-Based Demand Triggering**: Demands activate at exact scheduled times from CSV data
- **AI-Powered Optimization**: Uses Grok AI for intelligent vehicle allocation decisions
- **Real-time MCP Integration**: Fetches live traffic and weather data via APIs
- **Interactive Dashboard**: Gradio-based web interface with live map visualization
- **Location Agnostic**: Works for any geographic location automatically
- **Performance Tracking**: Real-time metrics and statistics

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys
Create a `.env` file in the project root:
```bash
# Google Maps API Key for traffic data
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# OpenWeather API Key for weather data
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Grok AI API Key for optimization recommendations
GROK_API_KEY=your_grok_api_key_here
```

### 3. Run the System
```bash
python fleet_optimizer.py
```

### 4. Access Dashboard
Open your browser to: http://localhost:7865

## 📊 System Components

### Core Files
- `fleet_optimizer.py` - Main application with Gradio interface
- `ai_agents.py` - AI agent implementations
- `data/fleet_vehicles.json` - Vehicle fleet data
- `data/demand_requests.csv` - Demand requests with timestamps

### Key Features
- **Timing-Based Processing**: Demands trigger at exact times from dataset
- **AI Optimization**: Considers distance, traffic, weather, and priority
- **Real-time Updates**: Live dashboard with allocation lines
- **Performance Metrics**: Vehicle utilization, assignment rates, earnings

## 🎯 How It Works

1. **Demand Triggering**: System reads demand requests and activates them at scheduled times
2. **MCP Data Fetching**: Retrieves real-time traffic and weather data for optimization
3. **AI Decision Making**: Grok AI analyzes all factors to select optimal vehicle
4. **Allocation**: Assigns vehicle to demand with visual feedback on map
5. **Tracking**: Updates performance metrics and statistics

## 📈 Performance Metrics

- Vehicle Utilization Rate
- Assignment Success Rate
- Average Response Time
- Total Earnings
- Real-time API Integration Status

## 🔧 Configuration

The system automatically detects location from your dataset coordinates and works for any geographic area. No manual configuration required!

## 🛠️ API Integration

- **Google Maps API**: Real-time traffic data
- **OpenWeather API**: Current weather conditions
- **Grok AI API**: Optimization recommendations

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
