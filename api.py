#!/usr/bin/env python3
"""
Vigilante IDS - REST API
Deploy on Render.com or any FastAPI‑compatible host.
"""
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Import our existing modules (they are synchronous)
from intrusion_detection.database import DatabaseManager
from intrusion_detection.auth import AuthManager
from intrusion_detection.model import IntrusionDetectionModel
from intrusion_detection.model_trainer import ModelTrainer
from intrusion_detection.utils import generate_pdf_report

# ---------- Pydantic Models ----------
class LoginRequest(BaseModel):
    username: str
    password: str

class OTPRequest(BaseModel):
    otp_code: str

class CreateUserRequest(BaseModel):
    username: str
    email: str
    role: str = "Analyst"

class ModifyUserRequest(BaseModel):
    username: str
    role: Optional[str] = None

class DeactivateUserRequest(BaseModel):
    username: str

class TrainRequest(BaseModel):
    input_file: str          # path on server (or you can upload file later)
    model_name: str
    threshold: float = 0.8
    features: Optional[List[str]] = None

class DetectRequest(BaseModel):
    input_file: str
    model_id: int
    explain: bool = False

class ReportRequest(BaseModel):
    period: str = "30d"
    output: Optional[str] = None

# ---------- Global state and lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.db = DatabaseManager()
    app.state.auth = AuthManager(app.state.db)
    app.state.trainer = ModelTrainer()
    print("✅ API started – Database connected")
    yield
    # Shutdown
    app.state.db.close()
    print("🛑 API shut down")

app = FastAPI(
    title="Vigilante IDS API",
    description="REST API for Intrusion Detection System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS – allow any origin for development, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ---------- Helper: run sync DB operations in thread ----------
async def run_sync(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

# ---------- Authentication Dependencies ----------
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    token = credentials.credentials
    auth: AuthManager = app.state.auth
    valid = await run_sync(auth.validate_session, token)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return auth.current_user

async def get_admin_user(current_user: Dict = Depends(get_current_user)):
    auth: AuthManager = app.state.auth
    if not auth.is_admin():
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return current_user

# ---------- Authentication Endpoints ----------
@app.post("/api/login")
async def login(req: LoginRequest):
    """Step 1: Verify credentials, send OTP."""
    auth: AuthManager = app.state.auth
    result = await run_sync(auth.login, req.username, req.password)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["message"])
    # If password change required
    if result.get("requires_password_change"):
        return {
            "success": False,
            "requires_password_change": True,
            "user_id": result.get("user_id"),
            "message": result["message"]
        }
    return {
        "success": True,
        "message": result["message"],
        "user_id": result.get("user_id"),
        "email": result.get("email"),
        "requires_otp": True
    }

@app.post("/api/verify-otp")
async def verify_otp(req: OTPRequest, user_state: dict = Depends(lambda: None)):
    """
    Step 2: Verify OTP and return session token.
    Note: The OTP secret is stored temporarily in the AuthManager instance.
    """
    auth: AuthManager = app.state.auth
    result = await run_sync(auth.verify_otp, req.otp_code)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["message"])
    return {
        "success": True,
        "session_token": result["session_token"],
        "username": result["username"],
        "role": result["role"],
        "user_id": result["user_id"]
    }

@app.post("/api/logout")
async def logout(current_user: Dict = Depends(get_current_user)):
    auth: AuthManager = app.state.auth
    await run_sync(auth.logout)
    return {"success": True, "message": "Logged out"}

@app.post("/api/change-password")
async def change_password(
    old_password: str,
    new_password: str,
    current_user: Dict = Depends(get_current_user)
):
    auth: AuthManager = app.state.auth
    result = await run_sync(
        auth.change_password,
        current_user["id"],
        old_password,
        new_password
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return {"success": True, "message": "Password changed"}

# ---------- Model Management ----------
@app.get("/api/models")
async def list_models(current_user: Dict = Depends(get_current_user)):
    db: DatabaseManager = app.state.db
    user_id = current_user["id"]
    if app.state.auth.is_admin():
        models = await run_sync(db.get_all_models)
    else:
        models = await run_sync(db.get_user_models, user_id)
    # Convert datetime objects to strings
    for m in models:
        if "created_at" in m and hasattr(m["created_at"], "isoformat"):
            m["created_at"] = m["created_at"].isoformat()
    return {"models": models}

@app.post("/api/train")
async def train_model(
    req: TrainRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Train a new RNSA+KNN model. The input_file must be accessible on the server."""
    if not app.state.auth.has_permission("train_models"):
        raise HTTPException(status_code=403, detail="Permission denied")
    if not os.path.exists(req.input_file):
        raise HTTPException(status_code=400, detail="Input file not found")
    try:
        result = await run_sync(
            app.state.trainer.train_model,
            data_path=req.input_file,
            model_name=req.model_name,
            r_s=0.01,
            max_detectors=1000,
            k=1,
            dataset_name=os.path.basename(req.input_file)
        )
        # Save to database
        db = app.state.db
        model_id = await run_sync(
            db.save_model,
            user_id=current_user["id"],
            model_name=result["model_name"],
            model_path=result["model_path"],
            dataset_name=os.path.basename(req.input_file),
            metrics=result["metrics"],
            features=result["feature_analysis"].get("available_features"),
            parameters=result["parameters"]
        )
        return {"success": True, "model_id": model_id, "metrics": result["metrics"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect")
async def detect(
    req: DetectRequest,
    current_user: Dict = Depends(get_current_user)
):
    if not app.state.auth.has_permission("run_detection"):
        raise HTTPException(status_code=403, detail="Permission denied")
    if not os.path.exists(req.input_file):
        raise HTTPException(status_code=400, detail="Input file not found")
    db = app.state.db
    model_data = await run_sync(db.get_model, req.model_id, current_user["id"])
    if not model_data:
        raise HTTPException(status_code=404, detail="Model not found")
    model_path = model_data["model_path"]
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file missing on server")
    # Load model
    model = IntrusionDetectionModel.load(model_path)
    # Run detection
    trainer = ModelTrainer()
    results = await run_sync(
        trainer.detect_anomalies,
        model_path=model_path,
        data_path=req.input_file,
        threshold=model.threshold
    )
    # Save detection record
    detection_id = await run_sync(
        db.save_detection,
        user_id=current_user["id"],
        model_id=req.model_id,
        input_file=req.input_file,
        results=results
    )
    return {
        "success": True,
        "detection_id": detection_id,
        "results": results
    }

@app.get("/api/detections")
async def get_detections(
    period: str = "30d",
    current_user: Dict = Depends(get_current_user)
):
    db = app.state.db
    days = int(period.rstrip("d"))
    if app.state.auth.is_admin():
        detections = await run_sync(db.get_all_detections, days)
    else:
        # For non-admin, get only user's detections
        summary = await run_sync(db.get_detection_summary, current_user["id"], days)
        # Also get full detection objects
        detections = await run_sync(db.get_detection_history, current_user["id"], 100)
    for d in detections:
        if "created_at" in d and hasattr(d["created_at"], "isoformat"):
            d["created_at"] = d["created_at"].isoformat()
    return {"detections": detections}

# ---------- Admin Only Endpoints ----------
@app.post("/api/admin/user-create")
async def admin_create_user(
    req: CreateUserRequest,
    _: Dict = Depends(get_admin_user)
):
    db = app.state.db
    auth = app.state.auth
    # Generate random temporary password
    import secrets
    temp_pass = secrets.token_urlsafe(8)
    password_hash = auth.hash_password(temp_pass)
    try:
        user_id = await run_sync(
            db.create_user,
            username=req.username,
            password_hash=password_hash,
            email=req.email,
            role=req.role,
            created_by=auth.current_user["id"]
        )
        return {
            "success": True,
            "user_id": user_id,
            "temporary_password": temp_pass,
            "message": f"User {req.username} created. Must change password on first login."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/admin/user-modify")
async def admin_modify_user(
    req: ModifyUserRequest,
    _: Dict = Depends(get_admin_user)
):
    db = app.state.db
    user = await run_sync(db.get_user, req.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if req.role:
        await run_sync(db.update_user_role, user["id"], req.role, app.state.auth.current_user["id"])
    return {"success": True, "message": f"User {req.username} updated"}

@app.post("/api/admin/user-deactivate")
async def admin_deactivate_user(
    req: DeactivateUserRequest,
    _: Dict = Depends(get_admin_user)
):
    db = app.state.db
    user = await run_sync(db.get_user, req.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Prevent deactivating the only admin
    if user["role_name"] == "Administrator":
        admin_count = await run_sync(db.count_admins)
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot deactivate the only administrator")
    await run_sync(db.deactivate_user, user["id"], app.state.auth.current_user["id"])
    await run_sync(db.invalidate_user_sessions, user["id"])
    return {"success": True, "message": f"User {req.username} deactivated"}

@app.get("/api/admin/audit-logs")
async def get_audit_logs(
    period: str = "30d",
    _: Dict = Depends(get_admin_user)
):
    db = app.state.db
    days = int(period.rstrip("d"))
    logs = await run_sync(db.get_audit_logs, days)
    for log in logs:
        if "created_at" in log and hasattr(log["created_at"], "isoformat"):
            log["created_at"] = log["created_at"].isoformat()
    return {"logs": logs}

@app.post("/api/admin/system-report")
async def generate_system_report(
    req: ReportRequest,
    _: Dict = Depends(get_admin_user)
):
    db = app.state.db
    days = int(req.period.rstrip("d"))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    # Collect data
    detection_summary = await run_sync(db.get_detection_summary, None, days)
    user_activity = await run_sync(db.get_user_activity, days)
    all_models = await run_sync(db.get_all_models)
    all_detections = await run_sync(db.get_all_detections, days)
    recent_anomalies = await run_sync(db.get_recent_anomalies, days, 20)
    total_flows = sum(d.get("total_flows", 0) for d in detection_summary)
    total_anomalies = sum(d.get("total_anomalies", 0) for d in detection_summary)
    report_data = {
        "report_period": {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "days": days
        },
        "detection_summary": {
            "total_flows_analyzed": total_flows,
            "total_anomalies_detected": total_anomalies,
            "detection_rate": total_anomalies / total_flows if total_flows else 0,
            "avg_false_positive_rate": 0.0  # simplified
        },
        "user_activity": user_activity,
        "recent_anomalies": recent_anomalies,
        "all_models": all_models,
        "all_detections": all_detections
    }
    # Generate PDF
    output_path = req.output or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = await run_sync(generate_pdf_report, report_data, output_path)
    # In a real scenario you would return the file; here we return the path.
    return {"success": True, "report_path": pdf_path}

# ---------- Health check ----------
@app.get("/api/health")
async def health():
    db: DatabaseManager = app.state.db
    ok = await run_sync(db.health_check)
    if ok:
        return {"status": "healthy", "database": "connected"}
    else:
        raise HTTPException(status_code=500, detail="Database unhealthy")

# ---------- Run directly ----------
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)