FROM python:3.7
WORKDIR /app
COPY requirements.txt ./requirements.txt 
RUN pip3 install -r requirements.txt
COPY . /app
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["head_pose_estimation_app.py"]