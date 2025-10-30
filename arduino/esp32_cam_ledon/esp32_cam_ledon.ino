// Simple program to turn on the built-in LED

const int LED_BUILTIN = 4;

void setup() {
  // Initialize the built-in LED pin as an output
  pinMode(LED_BUILTIN, OUTPUT);

  // Turn the LED on
  digitalWrite(LED_BUILTIN, HIGH);
  delay(1000);  
  digitalWrite(LED_BUILTIN, LOW);
  
}

void loop() {
  // Nothing to do here — LED stays on
}