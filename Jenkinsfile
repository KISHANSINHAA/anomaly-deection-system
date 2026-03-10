pipeline {
    agent any

    environment {
        IMAGE_NAME = "anomaly-detection-system"
        DOCKERHUB_USER = "kishansinha2207"
    }

    stages {

        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/KISHANSINHAA/anomaly-deection-system.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $DOCKERHUB_USER/$IMAGE_NAME:latest .'
            }
        }

        stage('Login DockerHub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-credentials',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh 'echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin'
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                sh 'docker push $DOCKERHUB_USER/$IMAGE_NAME:latest'
            }
        }

        stage('Deploy Container') {
            steps {
                sh 'docker stop anomaly_container || true'
                sh 'docker rm anomaly_container || true'
                sh 'docker run -d -p 8501:8501 --name anomaly_container $DOCKERHUB_USER/$IMAGE_NAME:latest'
            }
        }

    }
}
