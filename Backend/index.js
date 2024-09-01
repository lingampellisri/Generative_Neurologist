let express = require('express');
let app = express();
let allroutes = require('./allroutes');
const mongoose = require('mongoose');
const cors = require('cors');
const dotenv = require("dotenv");

dotenv.config();
app.use(express.json());


let corspolicy = {
    origin:"http://localhost:3000"
}
app.use(cors(corspolicy));


app.use((req,res,next) => {
    console.log(" Request received at " + (new Date()));
    next();
});

// connect
let db = async () => { 
    try{ 
        
        // console.log(process.env.DBURI);
        await mongoose.connect(process.env.DBURI);
        console.log(" connected to database");
    }
    catch(err) {
        console.log(' error connecting');
    }
}
db();



app.use('/',allroutes);

// connect to the database
// schema
// model
// from middleware, use model to get data from DB
const port = process.env.PORT || 3001; // Use the port provided by Vercel
app.listen(port, () => {
    console.log(`Backend server listening at port ${port}`);
});

// app.listen(3001,()=>{ console.log("Backend server listening at port 3001")});
