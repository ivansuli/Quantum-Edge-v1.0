# Quantum Edge — AI Trading Terminal

Sistema completo de trading algorítmico con IA que recibe señales de TradingView via Webhooks, las analiza con Machine Learning y aplica un motor de riesgo estricto antes de sugerir operaciones.

---

## Arquitectura del Sistema

```
TradingView Alert
       │
       ▼
┌─────────────────┐
│  FastAPI Server  │◄── Webhook Endpoint (POST /webhook)
│  (app.py)        │    Autenticación por secret + IP whitelist
└───────┬─────────┘
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│ Data Integrator  │────►│  APIs Externas   │
│                  │     │  • FRED (macro)   │
│                  │     │  • Alpha Vantage  │
│                  │     │  • Polygon (opts) │
│                  │     │  • Finnhub        │
└───────┬─────────┘     └──────────────────┘
        │
        ▼
┌─────────────────┐
│ Feature Engineer │  55+ features:
│                  │  • Tendencia (MA20/50/200, HH/HL)
│                  │  • Osciladores (RSI, MACD, ATR)
│                  │  • Volumen (VWAP, Delta, correlación)
│                  │  • S/R (pivot points, niveles psicológicos)
│                  │  • Macro (Fed, CPI, DXY, VIX)
│                  │  • Options Flow (call/put ratio)
│                  │  • Dark Pools (bloques institucionales)
│                  │  • Compuestas (confluencia, momentum)
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│ ML Engine        │  Ensemble:
│ (Predictive)     │  • Random Forest
│                  │  • Gradient Boosting
│                  │  • XGBoost
│                  │  → Probabilidad de éxito (Win Rate)
│                  │  → Time-Series Cross Validation
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│ Risk Manager     │  Reglas:
│                  │  • Max 1-2% riesgo por trade
│                  │  • R:R mínimo 1:2
│                  │  • Kelly Criterion (position sizing)
│                  │  • Correlación portfolio < 0.85
│                  │  • Detección zona resistencia
│                  │  • Límite drawdown diario 5%
│                  │  • Max posiciones abiertas
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│ Trade Decision   │──► Dashboard (HTML en tiempo real)
│ ✅ APPROVED      │──► Log en base de datos (SQLite)
│ ❌ REJECTED      │──► API REST para integraciones
└─────────────────┘
```

---

## Estructura de Archivos

```
trading-system/
├── app.py                          # FastAPI principal (servidor + endpoints + dashboard)
├── requirements.txt                # Dependencias Python
├── .env.example                    # Variables de entorno (copiar a .env)
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Configuración centralizada (Pydantic)
│
├── backend/
│   ├── __init__.py
│   ├── models.py                   # Modelos SQLAlchemy (Signal, Trade, Portfolio)
│   ├── risk_manager.py             # Motor de riesgo completo
│   ├── signal_processor.py         # Orquestador del pipeline
│   └── data_integrator.py          # Integración APIs externas
│
├── ml/
│   ├── __init__.py
│   ├── feature_engineering.py      # 55+ features (tendencia, vol, macro, etc.)
│   └── predictive_engine.py        # Ensemble ML (RF + GB + XGB)
│
├── templates/
│   └── dashboard.html              # UI del terminal de trading
│
├── models/                         # Modelos ML entrenados (auto-generado)
└── trading.db                      # Base de datos SQLite (auto-generado)
```

---

## Reglas Lógicas de IA (Risk Engine)

La IA **IGNORA** cualquier señal si:

1. **Probabilidad predictiva < 65%** — El modelo ML no tiene suficiente confianza
2. **Correlación > 0.85** — El activo está muy correlacionado con posiciones existentes
3. **Zona de resistencia sin volumen** — Precio en resistencia psicológica/técnica sin confirmación de volumen de ruptura
4. **Límite de drawdown diario (5%)** — Protección contra días catastróficos
5. **Máximo de posiciones alcanzado** — No se abren más trades si ya hay el máximo
6. **R:R < 2:1** — No se toman trades con mal ratio riesgo/beneficio

---

## Instalación Local

```bash
# 1. Clonar / copiar el proyecto
cd trading-system

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys

# 5. Ejecutar
python app.py
```

El sistema se inicia en `http://localhost:8000`

---

## Configuración en TradingView

### 1. Crear Alerta en TradingView

En TradingView, al crear una alerta, seleccionar **Webhook URL** y configurar:

- **URL**: `https://tu-dominio.com/webhook`
- **Método**: POST
- **Body (JSON)**:

```json
{
    "ticker": "{{ticker}}",
    "action": "BUY",
    "price": {{close}},
    "timeframe": "{{interval}}",
    "strategy": "Mi_Estrategia",
    "indicators": {
        "rsi": {{plot_0}},
        "macd": {{plot_1}},
        "volume_ratio": {{volume}} 
    },
    "secret": "tu-webhook-secret-aqui"
}
```

### 2. Variables de TradingView disponibles
- `{{ticker}}` — Símbolo (AAPL, MSFT, etc.)
- `{{close}}` — Precio de cierre
- `{{open}}`, `{{high}}`, `{{low}}` — OHLC
- `{{volume}}` — Volumen
- `{{interval}}` — Timeframe
- `{{plot_0}}`, `{{plot_1}}`, etc. — Valores de indicadores custom

---

## API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/webhook` | Recibir señal de TradingView |
| `GET` | `/api/status` | Estado del sistema + métricas ML |
| `GET` | `/api/signals` | Historial de señales procesadas |
| `GET` | `/api/portfolio` | Estado del portfolio actual |
| `POST` | `/api/train` | Re-entrenar modelo ML |
| `GET` | `/api/model/metrics` | Métricas del modelo ML |
| `POST` | `/api/settings` | Actualizar parámetros de riesgo |
| `POST` | `/api/simulate` | Simular señal (sin registrar) |
| `GET` | `/` | Dashboard HTML |

---

## Deploy en Hostinger

### Opción A: Hostinger VPS (Recomendado)

```bash
# 1. Conectarse al VPS por SSH
ssh root@tu-ip-hostinger

# 2. Instalar dependencias del sistema
apt update && apt install python3.11 python3.11-venv python3-pip nginx certbot -y

# 3. Crear usuario de la app
useradd -m -s /bin/bash trader
su - trader

# 4. Clonar proyecto
mkdir ~/quantum-edge && cd ~/quantum-edge
# Subir archivos via SCP o Git

# 5. Setup Python
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 6. Configurar .env
cp .env.example .env
nano .env  # Agregar tus API keys y cambiar SECRET_KEY

# 7. Crear servicio systemd
sudo tee /etc/systemd/system/quantum-edge.service << 'EOF'
[Unit]
Description=Quantum Edge Trading System
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/quantum-edge
Environment=PATH=/home/trader/quantum-edge/venv/bin
ExecStart=/home/trader/quantum-edge/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 8. Iniciar servicio
sudo systemctl daemon-reload
sudo systemctl enable quantum-edge
sudo systemctl start quantum-edge

# 9. Configurar Nginx como reverse proxy
sudo tee /etc/nginx/sites-available/quantum-edge << 'EOF'
server {
    listen 80;
    server_name tu-dominio.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/quantum-edge /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 10. SSL con Let's Encrypt
sudo certbot --nginx -d tu-dominio.com
```

### Opción B: Hostinger Web Hosting (Shared)

El hosting compartido de Hostinger soporta Python con algunas limitaciones. Pasos:

1. En el panel de Hostinger, ir a **Advanced → Python**
2. Crear aplicación Python con:
   - **Python Version**: 3.11+
   - **Application Root**: `/quantum-edge`
   - **Application Startup File**: `app.py`
   - **Application Entry Point**: `app` (el objeto FastAPI)
3. Subir archivos via File Manager o SSH
4. Instalar dependencias desde terminal SSH: `pip install -r requirements.txt`
5. Configurar el dominio para apuntar a la aplicación

### Variables de Entorno en Producción

Asegurarse de configurar en `.env`:

```env
DEBUG=false
SECRET_KEY=una-clave-segura-de-32-caracteres-minimo
WEBHOOK_SECRET=un-secret-unico-para-tradingview
```

---

## Features del Dashboard

- **KPIs en tiempo real**: Capital, señales, win rate, posiciones
- **Signal Feed**: Feed en vivo de todas las señales con detalles completos
- **Panel ML**: Métricas del modelo (AUC, Precisión, F1, CV)
- **Simulador**: Probar señales manualmente antes de conectar TradingView
- **Configuración**: Ajustar todos los parámetros de riesgo en vivo
- **Re-entrenamiento**: Reentrenar el modelo ML con un click
- **Auto-refresh**: Polling cada 5 segundos para nuevas señales

---

## Modelo ML — Detalles Técnicos

### Ensemble de 3 Modelos
1. **Random Forest** (300 árboles, max_depth=12)
2. **Gradient Boosting** (300 estimadores, lr=0.05)
3. **XGBoost** (300 rondas, max_depth=8)

### Voting: Soft (promedio de probabilidades)

### Cross Validation: TimeSeriesSplit (5 folds)
- Respeta la secuencia temporal (no hay data leakage)

### 55+ Features organizadas en 8 categorías:
1. Tendencia (MAs, slopes, HH/HL structure)
2. Osciladores (RSI + divergencias, MACD + cruces, ATR)
3. Volumen (VWAP, delta, correlación, breakout confirmation)
4. Soporte/Resistencia (pivot points, niveles psicológicos)
5. Macro (Fed rate, CPI, DXY, VIX, NFP)
6. Options Flow (call/put ratio, unusual activity)
7. Dark Pools (bloques institucionales, acumulación)
8. Compuestas (confluencia, momentum composite, risk-off)

### Position Sizing: Kelly Criterion (fraccionado al 50%)
- Optimiza el tamaño de posición según probabilidad de éxito y R:R

---

## Notas Importantes

- El modelo inicial se entrena con datos sintéticos. Para resultados reales, alimentarlo con historial de trades propios.
- Las APIs externas (FRED, Alpha Vantage, Polygon) requieren API keys gratuitas.
- El sistema es de **sugerencia** — la ejecución de trades debe ser manual o conectarse a un broker via API adicional.
- Siempre probar en paper trading antes de usar capital real.
- Este sistema no garantiza rentabilidad. El trading conlleva riesgos significativos.
