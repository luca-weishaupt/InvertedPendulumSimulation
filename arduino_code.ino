#include <PID_v1.h>
#include <Servo.h>
#include <SimpleKalmanFilter.h>

Servo myservo;

double Setpoint, Input, Output;
double Midpoint, Curpos, Posout;
int tolerance = 5;
double NewPos;
// gain level
double Ku = 0.2;
// period
double Pu = 3;
// PID params
double Kp = Ku;
double Ki = 2*Kp/Pu;
double Kd = 0;

PID myPID(&Input, &Output, &Setpoint, Kp, Ki, Kd, DIRECT);
PID posPID(&Curpos, &Posout, &Midpoint, 0.02, 0, 0, DIRECT);

SimpleKalmanFilter myKalmanFilter(5, 5, 0.001);
// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
  

  myservo.attach(9);
  Curpos = 90;
  myservo.write(Curpos);
  delay(5000);

  Input = analogRead(A0);
  Setpoint = 534;
  Midpoint = Curpos;
  myPID.SetMode(AUTOMATIC);
  myPID.SetOutputLimits(-90, 90);
  
  posPID.SetMode(AUTOMATIC);
  posPID.SetOutputLimits(-90, 90);
}

// the loop routine runs over and over again forever:
void loop() {
  // read the input on analog pin 0:
  Input = analogRead(A0);
//    Input = myKalmanFilter.updateEstimate(Input);
  Serial.print("Angle Input: ");
  Serial.print(Input);

  
  posPID.Compute();
    
  Serial.print(" Posout: ");
  Serial.print(Posout);
    
  NewPos = Curpos+Posout;
  
  if (abs(Input-Setpoint)>tolerance)
  {
    // compute new output:
    myPID.Compute();
    NewPos = NewPos+Output;
    Serial.print(" Output: ");
    Serial.print(Output);

  }
      
  if (NewPos<0){
    NewPos = 0;
  }
  if (NewPos>255){
    NewPos = 255;
  }
   
  Serial.print(" NewPos: ");
  Serial.println(NewPos);
   
  Curpos = NewPos;
  myservo.write(NewPos);
    
  delay(1);        // delay in between reads for stability
}
