from fastapi import APIRouter, Depends
from config.settings import settings, AppConfig, save_config
from core.security import verify_admin_key

router = APIRouter(prefix="/admin", tags=["Administration"])

@router.post("/config", dependencies=[Depends(verify_admin_key)])
def update_config(req: AppConfig):
    for key, value in req.dict().items():
        setattr(settings, key, value)
    save_config(settings)
    return {"status": "success", "config": settings.dict()}

@router.get("/config", dependencies=[Depends(verify_admin_key)])
def get_config():
    return settings.dict()