<b>Install Instruction:</b>

1) Install opencv-dev

2) Install pynder

3) Install numpy

4) Install caffe (you will want to find a tutorial for this one)


<b>Run Instructions:</b>

1) Log into tinder and prepare a data set

  a) Find your facebook id http://findmyfbid.com/ set fb_id to that as a string in CreateDB.py
  
  b) Go to https://www.facebook.com/dialog/oauth?client_id=464891386855067&redirect_uri=https://www.facebook.com/connect/login_success.html&scope=basic_info,email,public_profile,user_about_me,user_activities,user_birthday,user_education_history,user_friends,user_interests,user_likes,user_location,user_photos,user_relationship_details&response_type=token and log in. The page will redirect the browser twice. The first time will not update the page but the URL will change. Copy the new URL. It will then redirect again, to a blank.html page. You will have to do it all again if you miss the URL. The url will look like https://www.facebook.com/connect/login_success.html#access_token=CAAGm0PX4ZCpsBAIVasdfhakljsdfhlkahsdflakjsdhflkahdsvAFSDFLKJRH68HjnAxODI5daZBcg1K74BdaR9TiMD58Kz7pq5XvZAl6WFLdBhxTWAKY8WLfTNy2bFjHKo4Y8EWhDD5tlYeYJxwZCTRMi9HXIVLZAWwukcPyX5a4bfCMQqZC3QGO6ZCvLDalyen9iiLA2rQzt6t42qVZBZBzWjeiSTjLvZBgLM3TS78Hx9FUZD&expires_in=3926 Set fb_auth to the string of the access token.
  
  c) Run python CreateDB.py get [number of images] [latitude(o)] [longitude(o)]
  
  d) Run python CreateDB.py assess , left and right will record the swipes 
  
  e) If you need to completely reset everything you can call python CreateDB.py reset
  
  f) Run python CreateDB.py split [train fraction]
  
<b>2) Train the data set with caffe</b>

  a) Download caffe net for initial weights with ./get_caffenet.sh
  
  b) To start training use ~/caffe/build/tools/caffe train -solver=solver.prototxt -weights=caffenet.txt
  
  c) To resume training use ~/caffe/build/tools/caffe train -solver=solver.prototxt -snapshot=net_iter_#####.solverstate
  
  d) The accuracy reading from caffe will be low because rejects will be randomly assigned to 2 classes. Wait until the loss is low and stops steadly decreasing.
  
<b>3) Deploy the trained model on live Tinder data</b>

  a) Find your facebook id http://findmyfbid.com/ set fb_id to that as a string in TinderNet.py
  
  b) Go to https://www.facebook.com/dialog/oauth?client_id=464891386855067&redirect_uri=https://www.facebook.com/connect/login_success.html&scope=basic_info,email,public_profile,user_about_me,user_activities,user_birthday,user_education_history,user_friends,user_interests,user_likes,user_location,user_photos,user_relationship_details&response_type=token and log in. The page will redirect the browser twice. The first time will not update the page but the URL will change. Copy the new URL. It will then redirect again, to a blank.html page. You will have to do it all again if you miss the URL. The url will look like
  https://www.facebook.com/connect/login_success.html#access_token=CAAGm0PX4ZCpsBAIVasdfhakljsdfhlkahsdflakjsdhflkahdsvAFSDFLKJRH68HjnAxODI5daZBcg1K74BdaR9TiMD58Kz7pq5XvZAl6WFLdBhxTWAKY8WLfTNy2bFjHKo4Y8EWhDD5tlYeYJxwZCTRMi9HXIVLZAWwukcPyX5a4bfCMQqZC3QGO6ZCvLDalyen9iiLA2rQzt6t42qVZBZBzWjeiSTjLvZBgLM3TS78Hx9FUZD&expires_in=3926 Set fb_auth to the string of the access token.
  
  c) Run python TinderNet.py [iteration number] [lat] [long] [send swipes (true/false)]
  
  You can also leave out the lat and long if you want to use the most recent location
  
<b>4) Validation hasnt been written yet</b>
