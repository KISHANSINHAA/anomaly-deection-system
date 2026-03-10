pipeline {
    agent any

    environment {
        IMAGE_NAME = "anomaly-detection-system"
        DOCKERHUB_USER = "your_dockerhub_username"
    }

    stages {

        stage('Clone Repository') {
            steps {
                git branch: 'main',
                git 'https://github.com/KISHANSINHAA/anomaly-detection-system.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    sh 'docker build -t $DOCKERHUB_USER/$IMAGE_NAME:latest .'
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    withCredentials([string(credentialsId: 'dockerhub-password', variable: 'DOCKER_PASS')]) {
                        sh 'echo $DOCKER_PASS | docker login -u $DOCKERHUB_USER --password-stdin'
                        sh 'docker push $DOCKERHUB_USER/$IMAGE_NAME:latest'
                    }
                }
            }
        }

        stage('Deploy Container') {
            steps {
                script {
                    sh 'docker stop anomaly_container || true'
                    sh 'docker rm anomaly_container || true'
                    sh 'docker run -d -p 8501:8501 --name anomaly_container $DOCKERHUB_USER/$IMAGE_NAME:latest'
                }
            }
        }

    }
}
