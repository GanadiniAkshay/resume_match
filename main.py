from flask import Flask 
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class ResumeMain(Resource):
    def get(self):
        return {
                'About':"This is an API service for matching resumes to job descriptions",
                'Routes':[]
               }

api.add_resource(ResumeMain, '/')

if __name__ == '__main__':
    app.run(debug=True)