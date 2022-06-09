FROM python:3.7
WORKDIR /app
COPY requirements.txt ./requirements.txt 
RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY . /app
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["head_pose_estimation_app.py"]