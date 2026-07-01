```mermaid
flowchart LR

    USER([Company Ticker])

    EDGAR[(SEC EDGAR)]

    XBRL[XBRL Filings]

    INGEST["Ingestion Agent"]

    STATE[(Shared Financial State)]

    USER --> EDGAR
    EDGAR --> XBRL
    XBRL --> INGEST
    INGEST --> STATE

    classDef dark fill:#111111,stroke:#FFFFFF,color:#FFFFFF,stroke-width:2px;
    class USER,EDGAR,XBRL,INGEST,STATE dark;

    linkStyle default stroke:#FFFFFF,stroke-width:2px
```
