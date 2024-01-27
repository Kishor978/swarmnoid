#include "ESP8266WiFi.h"
#include "PubSubClient.h"

//watch monster moterdriver pinout  for ain1, ain2, bin1, bin2, etc
//enable pin should be high with 3.3v
#define m1_ain1 16 //D0 
#define m1_ain2 5   //D1
#define m1_bin1 4   //D2
#define m1_bin2 0   //D3
#define m1_pwma 14    //D5
#define m1_pwmb 12    //D6


const char* ssid = "Robotics";  // Enter SSID here
const char* password = "Robotics123"; 
const char* mqtt_server = "192.168.1.97";


WiFiClient espClient22;
PubSubClient client(espClient22);


void setup_wifi() {
  delay(10);
  
  Serial.println();
  
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("WiFi connected - NodeMCU IP address: ");
  Serial.println(WiFi.localIP());
}

// This functions is executed when some device publishes a message to a topic that your NodeMCU is subscribed to



void callback(String topic, byte* message, unsigned int length) {
  Serial.print("Message arrived on topic: ");
  Serial.print(topic);
  Serial.print(". Message: ");
  
  String messageInfo;
  for (int i = 0; i < length; i++) {
    Serial.print((char)message[i]);
    messageInfo += (char)message[i];
  }
  Serial.println();

  // If a message is received on the topic room/lamp, you check if the message is either on or off. Turns the lamp GPIO according to the message
  if(topic=="/robot"){
      Serial.print("Displaying message to users");
      // if(messageInfo == "allert"){
      //  len = messageInfo.length();
      //  d = 1;
      //  len = length;
      if(messageInfo == "forward"){
        forward();
      }
      if(messageInfo == "backward"){
        backward();
      }
      if(messageInfo == "left"){
        left();
      }
      if(messageInfo == "right"){
        right();
      }
      if(messageInfo == "stop"){
        stop();
      }
      //  Serial.print("allert");
      // delay(1000);
      
      // else if(messageInfo == "safe"){
      //   lcd.setCursor(2,0);
      //  lcd.print("safe  ");
      //   lcd.setCursor(2,1);
      //  lcd.print("safe  "); 
      //   Serial.print("safe");
      // }
  }
  Serial.println();
}
void backward()
{
  digitalWrite(m1_ain1, LOW);
  digitalWrite(m1_bin1, LOW);
  digitalWrite(m1_ain2, HIGH);
  digitalWrite(m1_bin2, HIGH);
  Serial.println("forward");
}

void forward()
{
  digitalWrite(m1_ain1, HIGH);
  digitalWrite(m1_bin1, HIGH);
  digitalWrite(m1_ain2, LOW);
  digitalWrite(m1_bin2, LOW);
  Serial.println("backward");
}

void left()
{
  digitalWrite(m1_ain1, LOW);
  digitalWrite(m1_bin1, HIGH);
  digitalWrite(m1_ain2, HIGH);
  digitalWrite(m1_bin2, LOW);
  Serial.println("left");
}

void right()
{
  digitalWrite(m1_ain1, HIGH);
  digitalWrite(m1_bin1, LOW);
  digitalWrite(m1_ain2, LOW);
  digitalWrite(m1_bin2, HIGH);
  Serial.println("right");
}

void stop()
{
  digitalWrite(m1_ain1, LOW);
  digitalWrite(m1_bin1, LOW);
  digitalWrite(m1_ain2, LOW);
  digitalWrite(m1_bin2, LOW);
  Serial.println("stop");
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    
    
    if (client.connect("ESP8266Client22")) {
      Serial.println("connected");  
      // Subscribe or resubscribe to a topic
      // You can subscribe to more topics (to control more LEDs in this example)
      client.subscribe("/robot");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}
void setup() {
  Serial.begin(115200);
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(m1_ain1, OUTPUT);
  pinMode(m1_ain2, OUTPUT);
  pinMode(m1_bin1, OUTPUT);
  pinMode(m1_bin2, OUTPUT);
  pinMode(m1_pwma, OUTPUT);
  pinMode(m1_pwmb, OUTPUT);
  digitalWrite(m1_pwma, HIGH);
  digitalWrite(m1_pwmb, HIGH);
}
void loop() {
  if (!client.connected()) {
    reconnect();
  }

if(!client.loop()){
    client.connect("ESP8266Client22");
      }
}
