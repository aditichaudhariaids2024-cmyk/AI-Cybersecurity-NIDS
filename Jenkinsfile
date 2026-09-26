pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                echo 'Checking out AI Cybersecurity NIDS project...'
                checkout scm
            }
        }

        stage('Docker Check') {
            steps {
                sh 'docker --version'
                sh 'docker ps'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building AI Cybersecurity NIDS Docker image...'
                sh 'docker build -t ai-cybersecurity-nids:${BUILD_NUMBER} .'
            }
        }

        stage('Build Success') {
            steps {
                echo 'AI Cybersecurity NIDS Docker image built successfully!'
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