pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                echo 'Checking out project code...'
                checkout scm
            }
        }

        stage('Environment Check') {
            steps {
                bat 'python --version'
                bat 'docker --version'
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running project tests...'
                bat 'python -m pytest -q'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image...'
                bat 'docker build -t ai-cybersecurity-nids:%BUILD_NUMBER% .'
            }
        }
    }

    post {
        success {
            echo 'CI Pipeline completed successfully!'
        }

        failure {
            echo 'CI Pipeline failed. Check the console output.'
        }
    }
}