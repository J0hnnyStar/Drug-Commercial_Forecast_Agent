# Drug Commercial Forecast Agent - System Architecture

```mermaid
graph LR
    %% UI
    User[User Query - Commercial forecast for Keytruda in Oncology]

    %% Control plane
    subgraph "Control"
        direction TB
        GPT5["GPT-5 Orchestrator<br/>gpt5_orchestrator.py"]
        Router["Model Router<br/>model_router.py"]
    end

    %% Multi-Agent System
    subgraph "Multi-Agent System"
        direction TB
        DataAgent["DataCollectionAgent<br/>DeepSeek V3.1"]
        MarketAgent["MarketAnalysisAgent<br/>GPT-5"]
        ForecastAgent["ForecastAgent<br/>Ensemble"]
        ReviewAgent["ReviewAgent<br/>Perplexity Sonar"]
    end

    %% Data Sources
    subgraph "Data Sources"
        direction TB
        FDA[(FDA Approvals)]
        SEC[(SEC EDGAR)]
        Clinical[(Clinical Trials)]
        Market[(Market Intel)]
    end

    %% Forecasting Methods
    subgraph "Forecasting Methods"
        direction TB
        Analog["Analogs 35%"]
        Bass["Bass Diffusion 25%"]
        Patient["Patient Flow 25%"]
        ML["ML Ensemble 15%"]
    end

    %% System Infrastructure
    subgraph "System Infrastructure"
        direction TB
        Monitor["System Monitor<br/>system_monitor.py"]
        TAP["TA Priors<br/>ta_priors.py"]
        Baselines["Baselines<br/>baselines.py"]
    end

    %% Validation
    subgraph "Validation"
        direction TB
        Historical["Historical Validation<br/>phase5_real_validation.py"]
        DataAudit["Data Audit<br/>phase5_data_audit.py"]
        RealData["Real Drug Data 114 Launches"]
    end

    %% Output
    Output["Final Output<br/>Peak Sales, Confidence, Audit Trail"]

    %% Primary Flow
    User --> GPT5
    GPT5 <--> Router

    %% Router to agents (provider choice)
    Router <--> DataAgent
    Router <--> MarketAgent
    Router <--> ReviewAgent

    %% Data collection fan-out
    DataAgent --> FDA
    DataAgent --> SEC
    DataAgent --> Clinical
    DataAgent --> Market

    %% Forecasting
    GPT5 --> ForecastAgent
    ForecastAgent --> Analog
    ForecastAgent --> Bass
    ForecastAgent --> Patient
    ForecastAgent --> ML

    %% Infrastructure support (dashed)
    GPT5 -.-> Monitor
    GPT5 -.-> TAP
    ForecastAgent -.-> TAP
    ForecastAgent -.-> Baselines

    %% Validation feedback (dashed back to control)
    Historical --> RealData
    DataAudit --> RealData
    RealData -.-> GPT5
    Historical -.-> GPT5

    %% Styling
    classDef orchestrator fill:#ff6b6b,stroke:#333,stroke-width:3px,color:#fff
    classDef agent fill:#4ecdc4,stroke:#333,stroke-width:2px,color:#fff
    classDef data fill:#45b7d1,stroke:#333,stroke-width:2px,color:#fff
    classDef method fill:#96ceb4,stroke:#333,stroke-width:2px,color:#333
    classDef system fill:#feca57,stroke:#333,stroke-width:2px,color:#333
    classDef validation fill:#ff9ff3,stroke:#333,stroke-width:2px,color:#333
    classDef output fill:#54a0ff,stroke:#333,stroke-width:3px,color:#fff

    class GPT5,Router orchestrator
    class DataAgent,MarketAgent,ForecastAgent,ReviewAgent agent
    class FDA,SEC,Clinical,Market data
    class Analog,Bass,Patient,ML method
    class Monitor,TAP,Baselines system
    class Historical,DataAudit,RealData validation
    class Output output
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

This architecture achieves near-consultant performance while maintaining cost efficiency and full transparency through comprehensive audit trailss.