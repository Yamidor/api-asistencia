-- Crear la base de datos
CREATE DATABASE IF NOT EXISTS attendance_system;
USE attendance_system;

-- Tabla de roles
CREATE TABLE roles (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL
);

-- Insertar roles básicos
INSERT INTO roles (name) VALUES 
('directivo'), 
('administrativo'), 
('estudiante'), 
('docente');

-- Tabla de grados (actualizada para incluir niveles de prescolar a once)
CREATE TABLE grades (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL,
    level VARCHAR(20) NOT NULL -- Agregamos nivel para categorizar
);

-- Insertar grados de prescolar a once
INSERT INTO grades (name, level) VALUES 
('Prejardín', 'prescolar'),
('Jardín', 'prescolar'),
('Transición', 'prescolar'),
('Primero', 'primaria'),
('Segundo', 'primaria'),
('Tercero', 'primaria'),
('Cuarto', 'primaria'),
('Quinto', 'primaria'),
('Sexto', 'secundaria'),
('Séptimo', 'secundaria'),
('Octavo', 'secundaria'),
('Noveno', 'secundaria'),
('Décimo', 'media'),
('Once', 'media');

-- Tabla de usuarios (actualizada para incluir número de documento)
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    document_number VARCHAR(20) UNIQUE NOT NULL, -- Nuevo campo de número de documento
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    role_id INT,
    grade_id INT NULL,
    face_encoding TEXT NOT NULL,
    face_image MEDIUMBLOB, -- Nuevo campo para almacenar la imagen
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (role_id) REFERENCES roles(id),
    FOREIGN KEY (grade_id) REFERENCES grades(id)
);

-- Tabla de asistencia
CREATE TABLE attendance (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    check_in TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);