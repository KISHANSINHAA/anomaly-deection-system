pipeline {
    agent any

    environment {
        IMAGE_NAME = "anomaly-detection-system"
        DOCKERHUB_USER = "kishansinha2207"
        DOCKER_CREDENTIAL_ID = "3f1d3a28-5856-4e19-a3bb-dee6ec9f8883"
    }

    stages {

        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/KISHANSINHAA/anomaly-detection-system.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat "docker build -t %DOCKERHUB_USER%/%IMAGE_NAME%:latest ."
            }
        }

        stage('Login to DockerHub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: "${DOCKER_CREDENTIAL_ID}",
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    bat "echo %DOCKER_PASS% | docker login -u %DOCKER_USER% --password-stdin"
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                bat "docker push %DOCKERHUB_USER%/%IMAGE_NAME%:latest"
            }
        }

        stage('Deploy Container') {
            steps {
                bat "docker stop anomaly_container || exit 0"
                bat "docker rm anomaly_container || exit 0"
                bat "docker run -d -p 8501:8501 --name anomaly_container %DOCKERHUB_USER%/%IMAGE_NAME%:latest"
            }
        }
    }

    post {
        success {
            echo 'Pipeline executed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs.'
        }
    }
}
