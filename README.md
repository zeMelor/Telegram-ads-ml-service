# Telegram-ads-ml-service
Description of my Hackathon Master of Telegram Ads: Development Edition (2026)
# 🏆 Ad Views Prediction Service (ML)

<!-- Tech Stack Badges -->
![Status](https://img.shields.io/badge/Status-Winner%20(1st%20Place)-gold)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Async-green)
![Telethon](https://img.shields.io/badge/Telethon-Async%20Parser-orange)

> **Победитель хакатона "Master of Telegram Ads: Development Edition" (2026)**
>
> **Результат:** 1 место | Best Metric (Cold Start): **1.75% MAPE**

## О проекте

Разработка  ML-сервиса для прогнозирования рекламных охватов (Ad Views) в Telegram-каналах.
Основной вызов хакатона — работа в условиях **«холодного старта»** (отсутствие исторических данных для новых каналов) и высокая гетероскедастичность данных (разброс просмотров от 100 до 1,000,000+).

**Моя роль:** ML-Engineer . 
Отвечал за проектирование пайплайна, разработку стратегии "Bridge45" и внедрение парсинга данных.

---

## Архитектура решения (Hybrid Router)

Из-за NDA исходный код закрыт. Ниже представлена схема работы сервиса, реализующая паттерн **Strategy Pattern** для ML-инференса.

Система использует **Hybrid Router**, который автоматически маршрутизирует запросы в зависимости от наличия канала в базе знаний.

```mermaid
graph TD
    User[User / Client] -->|POST /predict| API[FastAPI Endpoint]
    API --> Router{Router Logic}
    
    %% Ветка исторических данных
    Router -->|Channel in Train DB? YES| Task1[Task 1 Pipeline]
    Task1 --> Stack[Ensemble: CatBoost + LGBM]
    Stack --> Res1[Prediction]
    
    %% Ветка новых каналов
    Router -->|Channel in Train DB? NO| Task2[Task 2 Pipeline]
    Task2 --> ETL[Telethon Parser]
    ETL -->|Real-time Metadata| Feat[Feature Engineering]
    Feat --> Bridge[Bridge45 Strategy]
    
    subgraph "Strategy: Bridge45 (Segmentation)"
    Bridge --> M1[Model: DeltaTG / Small]
    Bridge --> M2[Model: LogRMSE / Large]
    Bridge --> Weight{Dynamic Weighting}
    M1 --> Weight
    M2 --> Weight
    end
    
    Weight --> Res2[Prediction]
    
    Res1 --> Final[JSON Response]
    Res2 --> Final
