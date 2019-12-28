//import java.util.ArrayList;
var express = require('express');
var router = express.Router();

var ejs = require('ejs');
var http = require('http');
var fs = require('fs');

var filelist_;
var modedata_;

var nodemailer = require('nodemailer');
var fromid = 'hsw9494@gmail.com';
var frompw = 'tkddnjs789';
var toid = 'hsw9494@gmail.com';

var transporter = nodemailer.createTransport({
    service:'gmail',
    auth: {
        user : fromid,
        pass : frompw
    }
});

var mailOption = {
    from : fromid,
    to : toid,
    subject : 'nodemailer test',
    text : 'Hello'
};

//text 읽기
fs.readFile('./mode.txt', function (err, data) {
    if (err) throw err;
    modedata_ = data;
});

// 이미지 읽기
fs.readdir('./public/images', function(error, filelist){
    filelist_ = filelist
})

router.use(express.json());

//홈
router.get('/', function(req, res, next) {    
  res.render('index', { 
          bbb : filelist_ ,
          aaa : modedata_
  });
});


//모드 버튼값 받기 ModeWrite()
router.post('/ModeWrite', function(req,res){

    fs.writeFile('./mode.txt', req.body.mode, (err) => { 
        if (err) throw err;
        console.log("Write text success : " + req.body.mode );
    });
    res.send(null);
})

//모드 값 전달 ModeRead()
router.post('/ModeRead', function(req, res){
    fs.readFile('./mode.txt', function (err, data) {
        if (err) throw err;
        res.send(data)
    });
});


module.exports = router;
