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
from typing import Dict, List
from datetime import datetime, timedelta
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


class NonWorkingDay(Base):
    __tablename__ = "non_working_days"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False)
    description = Column(String(200), nullable=False)
    type = Column(String(50), nullable=False)  # 'holiday', 'vacation', 'special'
    created_at = Column(DateTime, default=lambda: datetime.now(colombia_tz))

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

# Función auxiliar para verificar si un día es laborable
def is_working_day(date: datetime, non_working_days: List[datetime]) -> bool:
    # Verificar si es fin de semana (5 = sábado, 6 = domingo)
    if date.weekday() in [5, 6]:
        return False
    
    # Verificar si es un día no laborable (festivo, vacación, etc.)
    date_without_time = date.date()
    if any(d.date() == date_without_time for d in non_working_days):
        return False
    
    return True

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
                    "document_number": user.document_number,
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

# Modificar la función get_attendance_by_date_range
@app.get("/attendance/")
async def get_attendance_by_date_range(
    start_date: str,
    end_date: str,
    db: Session = Depends(get_db)
):
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    # Obtener todos los días no laborables en el rango
    non_working_days = db.query(NonWorkingDay.date).filter(
        NonWorkingDay.date >= start,
        NonWorkingDay.date <= end
    ).all()
    non_working_dates = [d.date for d in non_working_days]

    # Calcular días hábiles en el rango
    def count_working_days(start_date, end_date, non_working_dates):
        current = start_date
        working_days = 0
        while current <= end_date:
            if is_working_day(current, non_working_dates):
                working_days += 1
            current += timedelta(days=1)
        return working_days

    total_working_days = count_working_days(start, end, non_working_dates)
    
    # Obtener el total de usuarios
    total_users = db.query(User).count()
    
    # Obtener usuarios únicos que registraron asistencia en días hábiles
    present_users_query = db.query(
        func.count(func.distinct(Attendance.user_id))
    ).filter(
        Attendance.check_in >= start,
        Attendance.check_in <= end
    )
    present_users = present_users_query.scalar() or 0
    
    # Calcular ausentes
    absent_users = total_users - present_users

    # Obtener todos los usuarios
    users = db.query(User).all()
    
    # Obtener todas las asistencias para el rango de fechas
    all_attendances = db.query(Attendance).filter(
        Attendance.check_in >= start,
        Attendance.check_in <= end
    ).all()

    # Crear un diccionario de asistencias por usuario
    attendance_by_user = {}
    for attendance in all_attendances:
        # Solo contar asistencias en días hábiles
        if is_working_day(attendance.check_in, non_working_dates):
            if attendance.user_id not in attendance_by_user:
                attendance_by_user[attendance.user_id] = []
            attendance_by_user[attendance.user_id].append(attendance.check_in.date())

    # Organizar usuarios por rol con información de inasistencias
    attendance_by_role = {}
    
    for user in users:
        user_attendances = attendance_by_user.get(user.id, [])
        days_attended = len(set(user_attendances))  # Usar set para contar días únicos
        days_absent = total_working_days - days_attended
        
        # Solo incluir usuarios con inasistencias en días hábiles
        if days_absent > 0:
            if user.role_id not in attendance_by_role:
                attendance_by_role[user.role_id] = []
            
            last_attendance = max(user_attendances) if user_attendances else None
            
            user_data = {
                "id": user.id,
                "document_number": user.document_number,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
                "role_id": user.role_id,
                "grade_id": user.grade_id,
                "daysAbsent": days_absent,
                "check_in": last_attendance
            }
            
            attendance_by_role[user.role_id].append(user_data)
    
    return {
        "stats": {
            "total": total_users,
            "present": present_users,
            "absent": absent_users,
            "total_working_days": total_working_days
        },
        "attendance_by_role": attendance_by_role
    }

# Nueva ruta para gestionar días no laborables
@app.post("/non-working-days/")
async def create_non_working_day(
    date: str = Form(...),
    description: str = Form(...),
    type: str = Form(...),
    db: Session = Depends(get_db)
):
    print(f"Received data: date={date}, description={description}, type={type}")
    
    try:
        non_working_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
    
    db_non_working_day = NonWorkingDay(
        date=non_working_date,
        description=description,
        type=type
    )
    db.add(db_non_working_day)
    db.commit()
    return {"message": "Non-working day created successfully"}

@app.get("/non-working-days/")
async def get_non_working_days(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(NonWorkingDay)
    
    if start_date and end_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            query = query.filter(NonWorkingDay.date >= start, NonWorkingDay.date <= end)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
    
    return query.all()

@app.delete("/non-working-days/{day_id}")
async def delete_non_working_day(
    day_id: int,
    db: Session = Depends(get_db)
):
    non_working_day = db.query(NonWorkingDay).filter(NonWorkingDay.id == day_id).first()
    if not non_working_day:
        raise HTTPException(status_code=404, detail="Day not found")
    
    db.delete(non_working_day)
    db.commit()
    return {"message": "Non-working day deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)