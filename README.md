**ML service**
 - This ML service predict make of vehicle based on parking violations
 
*Notes*
 - Commands to build docker image.
     ```utility/commands```
 - Commands for kubernetes deployment
     ```deployment.yml```


- API Endpoint     - ```http://127.0.0.1:5000/prediction?Violation code=80.69BS&Route=00557&Body Style=PA&Agency=55```
- Response in JSON - ```{"TOYT":0.17274553157015307,"code":200,"status":"Success"}```