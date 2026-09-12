# AtlasAI
A custom AI travel agent designed to plan, coordinate, and optimize future family vacations based on shared preferences, budgets, and schedules.
```
+-------------------------------------------------------------+
|                     User Interface                          |
|         (Streamlit Web App / Telegram Bot / Chainlit)       |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     Orchestrator Agent                      |
|            (LangGraph / LlamaIndex / CrewAI)                |
+--------------+-------------------------------+--------------+
               |                               |
               v                               v
+------------------------------+ +----------------------------+
|        Memory Engine         | |        Search Tools        |
|  (User Profile DB + Vector   | | (Tavily Search / Google    |
|   Store / Fact Extraction)   | |  Places API / Serper)      |
+------------------------------+ +----------------------------+
```