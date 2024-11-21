from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, LargeBinary, func
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import face_recognition
import numpy as np
from datetime import datetime
import cv2
import json
from typing import Optional
from pydantic import BaseModel, EmailStr
import pytz
import io

# Configurar la zona horaria de Colombia
colombia_tz = pytz.timezone('America/Bogota')

# Configuración de la base de datos
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:12345@localhost/attendance_system"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelos SQLAlchemy (actualizados)
class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)

class Grade(Base):
    __tablename__ = "grades"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    level = Column(String(20), nullable=False)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    document_number = Column(String(20), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id"))
    grade_id = Column(Integer, ForeignKey("grades.id"), nullable=True)
    face_encoding = Column(Text, nullable=False)
    face_image = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(colombia_tz))

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    check_in = Column(DateTime, default=lambda: datetime.now(colombia_tz))

# Modelos Pydantic
class UserCreate(BaseModel):
    document_number: str
    first_name: str
    last_name: str
    email: EmailStr
    role_id: int
    grade_id: Optional[int] = None

class UserResponse(BaseModel):
    id: int
    document_number: str
    first_name: str
    last_name: str
    email: str
    role_id: int
    grade_id: Optional[int]

    class Config:
        from_attributes = True

# Inicialización de FastAPI
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Función auxiliar para obtener la fecha actual en Colombia
def get_current_colombia_time():
    return datetime.now(colombia_tz)

# Ruta para verificar si el documento ya existe
@app.get("/check-document/{document_number}")
async def check_document(document_number: str, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.document_number == document_number).first()
    return {"exists": existing_user is not None}

@app.post("/users/", response_model=UserResponse)
async def create_user(
    user: str = Form(...),
    face_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Convertir el string JSON a un diccionario
    user_data = json.loads(user)
    # Validar los datos usando Pydantic
    user_obj = UserCreate(**user_data)
    
    # Verificar si el email o documento ya existen
    existing_user = (
        db.query(User)
        .filter((User.email == user_obj.email) | (User.document_number == user_obj.document_number))
        .first()
    )
    if existing_user:
        raise HTTPException(status_code=400, detail="Email or document number already registered")
    
    # Procesar imagen y obtener encoding facial
    contents = await face_image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    face_locations = face_recognition.face_locations(img)
    if not face_locations:
        raise HTTPException(status_code=400, detail="No face detected in image")
    
    face_encoding = face_recognition.face_encodings(img, face_locations)[0]
    
    # Crear usuario en la base de datos
    db_user = User(
        document_number=user_obj.document_number,
        first_name=user_obj.first_name,
        last_name=user_obj.last_name,
        email=user_obj.email,
        role_id=user_obj.role_id,
        grade_id=user_obj.grade_id,
        face_encoding=json.dumps(face_encoding.tolist()),
        face_image=contents  # Guardar imagen original
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/recognize/")
async def recognize_face(
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Procesar imagen recibida
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detectar rostro en la imagen
    face_locations = face_recognition.face_locations(img)
    if not face_locations:
        raise HTTPException(status_code=400, detail="No face detected in image")
    
    unknown_encoding = face_recognition.face_encodings(img, face_locations)[0]
    
    # Obtener todos los usuarios
    users = db.query(User).all()
    
    for user in users:
        known_encoding = np.array(json.loads(user.face_encoding))
        # Comparar rostros
        matches = face_recognition.compare_faces([known_encoding], unknown_encoding)
        
        if matches[0]:
            # Verificar si ya registró asistencia hoy
            current_time = get_current_colombia_time()
            today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = current_time.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            existing_attendance = db.query(Attendance).filter(
                Attendance.user_id == user.id,
                Attendance.check_in >= today_start,
                Attendance.check_in <= today_end
            ).first()
            
            if existing_attendance:
                return {
                    "user_id": user.id,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "email": user.email,
                    "role_id": user.role_id,
                    "grade_id": user.grade_id,
                    "check_in": existing_attendance.check_in,
                    "already_registered": True,
                    # Convertir imagen a base64 para enviar al frontend
                    "face_image": f"data:image/jpeg;base64,{user.face_image.hex()}" if user.face_image else None
                }
            
            # Registrar nueva asistencia
            attendance = Attendance(user_id=user.id)
            db.add(attendance)
            db.commit()
            
            return {
                "user_id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
                "role_id": user.role_id,
                "grade_id": user.grade_id,
                "check_in": attendance.check_in,
                "already_registered": False,
                # Convertir imagen a base64 para enviar al frontend
                "face_image": f"data:image/jpeg;base64,{user.face_image.hex()}" if user.face_image else None
            }
    
    raise HTTPException(status_code=404, detail="Face not recognized")

# Rutas adicionales (las anteriores se mantienen igual)
@app.get("/grades/levels")
def get_grades_by_level(db: Session = Depends(get_db)):
    levels = db.query(Grade.level).distinct().all()
    grades_by_level = {}
    for level in levels:
        grades_by_level[level[0]] = db.query(Grade).filter(Grade.level == level[0]).all()
    return grades_by_level

if __name__ == "__main__":
    import uvicorn
    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)