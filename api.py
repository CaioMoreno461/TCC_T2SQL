from fastapi import FastAPI, HTTPException
from deep_translator import GoogleTranslator
import asyncpg
import sqlparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Configuração do FastAPI
app = FastAPI()

# Configuração de logs
logging.basicConfig(level=logging.INFO)

# Conexão com o banco de dados PostgreSQL
DB_CONNECTION = {
    'user': 'ttsql_owner',
    'password': '54ViGZBSuELt',
    'database': 'ttsql',
    'host': 'ep-red-recipe-a5wqkops.us-east-2.aws.neon.tech',
    'port': 5432,
    'ssl': 'require'
}

# Carregar o modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained("chatdb/natural-sql-7b")
model = AutoModelForCausalLM.from_pretrained("chatdb/natural-sql-7b")

# Tradutor Google
tradutor = GoogleTranslator(source='pt', target='en')

# Definição do prompt específico para o modelo Natural-SQL-7b
prompt_template = """
# Task
Generate a SQL query to answer the following question: `{question}`

### PostgreSQL Database Schema
The query will run on a database with the following schema:

CREATE TABLE patients (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100),
  gender VARCHAR(10),
  birthday DATE,
  contact_number VARCHAR(20),
  email VARCHAR(100),
  address VARCHAR(150),
  city_id INTEGER,
  enabled BOOLEAN
);

CREATE TABLE admissions (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER,
  admission_date TIMESTAMP,
  discharge_date TIMESTAMP,
  admission_reason VARCHAR(200),
  hospital_id INTEGER,
  admission_status VARCHAR(50),
  treatment_outcome VARCHAR(100),
  complications VARCHAR(200),
  bed_id INTEGER,
  department_id INTEGER,
  cid_primary INTEGER,
  FOREIGN KEY (patient_id) REFERENCES patients(id),
  FOREIGN KEY (hospital_id) REFERENCES hospitals(id),
  FOREIGN KEY (department_id) REFERENCES departments(id),
  FOREIGN KEY (cid_primary) REFERENCES cid_codes(id)
);

CREATE TABLE hospitals (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100),
  location VARCHAR(150),
  phone_number VARCHAR(20)
);

CREATE TABLE beds (
  id SERIAL PRIMARY KEY,
  hospital_id INTEGER,
  bed_type VARCHAR(50),
  availability_status VARCHAR(50),
  accomodation_type VARCHAR(50),
  FOREIGN KEY (hospital_id) REFERENCES hospitals(id)
);

CREATE TABLE treatments (
  id SERIAL PRIMARY KEY,
  admission_id INTEGER,
  treatment_description TEXT,
  start_date DATE,
  end_date DATE,
  doctor_id INTEGER,
  FOREIGN KEY (admission_id) REFERENCES admissions(id),
  FOREIGN KEY (doctor_id) REFERENCES users(id)
);

CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100),
  role VARCHAR(50),
  email VARCHAR(100),
  enabled BOOLEAN,
  chat_enabled BOOLEAN
);

CREATE TABLE exams (
  id SERIAL PRIMARY KEY,
  admission_id INTEGER,
  exam_type VARCHAR(100),
  exam_date DATE,
  results TEXT,
  FOREIGN KEY (admission_id) REFERENCES admissions(id)
);

CREATE TABLE patient_history (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER,
  history_description TEXT,
  date_recorded DATE,
  FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE departments (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100),
  description TEXT
);

CREATE TABLE medications (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100),
  dosage VARCHAR(50),
  administration_method VARCHAR(50)
);

CREATE TABLE treatment_medications (
  id SERIAL PRIMARY KEY,
  treatment_id INTEGER,
  medication_id INTEGER,
  dosage VARCHAR(50),
  frequency VARCHAR(50),
  start_date TIMESTAMP,
  end_date TIMESTAMP,
  FOREIGN KEY (treatment_id) REFERENCES treatments(id),
  FOREIGN KEY (medication_id) REFERENCES medications(id)
);

CREATE TABLE cid_codes (
  id SERIAL PRIMARY KEY,
  code VARCHAR(10),
  description TEXT
);

### SQL
Here is the SQL query that answers the question: `{question}`
'''sql
"""

# Função para executar a query SQL
async def execute_query(query):
    try:
        conn = await asyncpg.connect(**DB_CONNECTION)
        try:
            result = await conn.fetch(query)
            return result
        finally:
            await conn.close()
    except Exception as e:
        logging.error(f"Erro ao executar consulta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao executar consulta: {str(e)}")

# Função para gerar a SQL a partir da pergunta em inglês
def generate_sql(question_en):
    prompt = prompt_template.format(question=question_en)
    inputs = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
        temperature=0.0,
        top_p=1,
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if "'''sql" in outputs[0]:
        sql_query = outputs[0].split("'''sql")[1].split("'''")[0]
        return sqlparse.format(sql_query, reindent=True)
    else:
        raise HTTPException(status_code=400, detail="Erro ao gerar consulta SQL.")

# Função para traduzir pergunta para inglês
def traduzir_pergunta(pergunta_pt):
    return tradutor.translate(pergunta_pt)

# Rota principal para receber perguntas e gerar respostas
@app.post("/query/")
async def process_question(pergunta_pt: str):
    try:
        # Validar a entrada
        if not pergunta_pt.strip():
            raise HTTPException(status_code=400, detail="Pergunta inválida.")
        
        # Traduzir a pergunta para o inglês
        pergunta_en = traduzir_pergunta(pergunta_pt)

        # Gerar SQL a partir da pergunta traduzida
        sql_query = generate_sql(pergunta_en)

        # Executar a consulta no banco de dados
        resultado = await execute_query(sql_query)

        # Se for uma tabela (lista de tuplas), retornar uma resposta adequada
        if isinstance(resultado, list):
            return {"response": "Aqui está o resultado da sua consulta.", "tabela": [dict(row) for row in resultado]}

        return {"response": resultado}
    except Exception as e:
        logging.error(f"Erro no processamento: {str(e)}")
        return {"response": "Não consigo responder sua pergunta no momento...", "error": str(e)}
