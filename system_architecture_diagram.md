# Drug Commercial Forecast Agent - System Architecture

```mermaid
graph TB
    %% User Interface Layer
    User[👤 User Query<br/>"Commercial forecast for<br/>Keytruda in Oncology"]
    
    %% Main Orchestrator
    GPT5[🎯 GPT-5 Orchestrator<br/>gpt5_orchestrator.py<br/>8-Step Pipeline Controller]
    
    %% Model Router
    Router[🔀 Model Router<br/>model_router.py<br/>Task-Based LLM Routing]
    
    %% Specialized Agents
    subgraph "Multi-Agent System"
        DataAgent[📊 DataCollectionAgent<br/>DeepSeek V3.1<br/>Bulk Data Processing]
        MarketAgent[📈 MarketAnalysisAgent<br/>GPT-5<br/>Complex Reasoning]
        ForecastAgent[🔮 ForecastAgent<br/>Multi-Method<br/>Ensemble Forecasting]
        ReviewAgent[🔍 ReviewAgent<br/>Perplexity Sonar<br/>Objective Critique]
    end
    
    %% Data Sources
    subgraph "Data Sources"
        FDA[(🏛️ FDA<br/>Approvals<br/>Mechanisms)]
        SEC[(💰 SEC EDGAR<br/>Revenue Data<br/>10-K/10-Q)]
        Clinical[(🧪 Clinical Trials<br/>Efficacy<br/>Safety)]
        Market[(📊 Market Intel<br/>Competition<br/>Pricing)]
    end
    
    %% Forecasting Methods
    subgraph "Forecasting Methods"
        Analog[🔄 Analog Projection<br/>35% weight<br/>Industry Standard]
        Bass[📈 Bass Diffusion<br/>25% weight<br/>Adoption Curves]
        Patient[👥 Patient Flow<br/>25% weight<br/>Market Sizing]
        ML[🤖 ML Ensemble<br/>15% weight<br/>TA-Calibrated]
    end
    
    %% System Components
    subgraph "System Infrastructure"
        Monitor[📊 System Monitor<br/>system_monitor.py<br/>Audit Trail]
        TAP[⚙️ TA Priors<br/>ta_priors.py<br/>Therapeutic Area<br/>Parameters]
        Baselines[📏 Baselines<br/>baselines.py<br/>Industry Benchmarks]
    end
    
    %% Validation System
    subgraph "Validation System"
        Historical[📋 Historical Validation<br/>phase5_real_validation.py<br/>Real LLM Calls]
        DataAudit[🔍 Data Audit<br/>phase5_data_audit.py<br/>Quality Checks]
        RealData[💾 Real Drug Data<br/>114 Launches<br/>Keytruda, Repatha]
    end
    
    %% Output
    Output[📊 Final Output<br/>Peak Sales Forecast<br/>Confidence Score<br/>Audit Trail]
    
    %% Flow Connections
    User --> GPT5
    
    %% Step 1: Query Parsing
    GPT5 -->|Step 1: Parse Query| Router
    Router -->|GPT-5| GPT5
    
    %% Step 2: Data Collection
    GPT5 -->|Step 2: Orchestrate| DataAgent
    DataAgent --> Router
    Router -->|DeepSeek| DataAgent
    DataAgent --> FDA
    DataAgent --> SEC
    DataAgent --> Clinical
    DataAgent --> Market
    
    %% Step 3: Data Review
    GPT5 -->|Step 3: Review Quality| ReviewAgent
    ReviewAgent --> Router
    Router -->|Perplexity| ReviewAgent
    
    %% Step 4: Market Analysis
    GPT5 -->|Step 4: Analyze Market| MarketAgent
    MarketAgent --> Router
    Router -->|GPT-5| MarketAgent
    
    %% Step 5: Multi-Method Forecast
    GPT5 -->|Step 5: Generate Forecasts| ForecastAgent
    ForecastAgent --> Analog
    ForecastAgent --> Bass
    ForecastAgent --> Patient
    ForecastAgent --> ML
    
    %% Step 6: Harsh Review
    GPT5 -->|Step 6: Critique| ReviewAgent
    
    %% Step 7-8: Ensemble & Validation
    GPT5 -->|Step 7-8: Ensemble| Output
    
    %% System Infrastructure Connections
    GPT5 --> Monitor
    GPT5 --> TAP
    ForecastAgent --> TAP
    
    %% Validation Connections
    Historical --> GPT5
    Historical --> RealData
    DataAudit --> RealData
    
    %% Styling
    classDef orchestrator fill:#ff6b6b,stroke:#333,stroke-width:3px,color:#fff
    classDef agent fill:#4ecdc4,stroke:#333,stroke-width:2px,color:#fff
    classDef data fill:#45b7d1,stroke:#333,stroke-width:2px,color:#fff
    classDef method fill:#96ceb4,stroke:#333,stroke-width:2px,color:#333
    classDef system fill:#feca57,stroke:#333,stroke-width:2px,color:#333
    classDef validation fill:#ff9ff3,stroke:#333,stroke-width:2px,color:#333
    classDef output fill:#54a0ff,stroke:#333,stroke-width:3px,color:#fff
    
    class GPT5 orchestrator
    class DataAgent,MarketAgent,ForecastAgent,ReviewAgent agent
    class FDA,SEC,Clinical,Market data
    class Analog,Bass,Patient,ML method
    class Monitor,TAP,Baselines system
    class Historical,DataAudit,RealData validation
    class Router,Output output
```

## Performance Metrics

### Achievement Status (Phase 5 Complete)
- ✅ **Multi-agent system**: **41.3% MAPE** (approaching 40% consultant baseline)
- ✅ **Peak heuristic baseline**: 71.2% MAPE (traditional method)
- ✅ **Ensemble baseline**: 80.8% MAPE (academic approach)
- ✅ **Drug differentiation**: Keytruda (34.6% APE), Repatha (33.6% APE)
- ✅ **Cost efficiency**: ~$0.16 per forecast vs $2M consultant cost

### Key Architecture Principles

**1. Linus-Style Data Structure Focus**
- Core data flows through AgentHierarchy and AgentTask structures
- "Data structures over algorithms" - everything defined by task routing

**2. Multi-Agent Specialization**
- Each agent does one thing well (following Unix philosophy)
- GPT-5 for orchestration and complex reasoning
- DeepSeek for bulk processing efficiency
- Perplexity for objective review with citations

**3. Production-Grade Features**
- Complete audit trails and decision logging
- Therapeutic area-specific calibration
- Real-time cost tracking and budget controls
- Fallback mechanisms for provider failures

**4. Validation-First Approach**
- Tests against real historical outcomes
- Multiple baseline comparisons
- Bootstrap confidence intervals
- Temporal evaluation framework

This architecture achieves near-consultant performance while maintaining cost efficiency and full transparency through comprehensive audit trails.