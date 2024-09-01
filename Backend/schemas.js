let mongoose= require('mongoose');
let userSchema = new mongoose.Schema(
    {
      "username": {
        "type": "String",
        "required": true
      },
      "password": {
        "type": "String",
        "required": true
      },
      "email": {
        "type": "String",
        "required": true,
        "unique":true
      },
      "phone": {
        "type": "String",
        "required": true
      },
    
    }
    );

    let usersModel = mongoose.model('User',userSchema);
    module.exports = {usersModel};