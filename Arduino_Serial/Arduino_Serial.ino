const int wPin = 8;
const int redPin = 11;
const int greenPin = 12;
const int bluePin = 13;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(wPin, OUTPUT);
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available() > 0){
    String msg = Serial.readString();

    if(msg == "Happy"){
      digitalWrite(greenPin, HIGH);
      delay(2000);
      digitalWrite(greenPin, LOW);   
    }
    else if (msg == "Surprise"){
      digitalWrite(bluePin, HIGH);
      delay(2000);
      digitalWrite(bluePin, LOW); 
    } 
    else if (msg == "Sad"){
      digitalWrite(redPin, HIGH);
      delay(2000);
      digitalWrite(redPin, LOW); 
    }
    else if (msg == "Neutral"){
      digitalWrite(wPin, HIGH);
      delay(2000);
      digitalWrite(wPin, LOW); 
    }
    else{
      for(int i = 0; i<5; i++){
          digitalWrite(bluePin, HIGH);  
          delay(100);                       
          digitalWrite(bluePin, LOW);   
          delay(100);
        } 
      }
    }
}
