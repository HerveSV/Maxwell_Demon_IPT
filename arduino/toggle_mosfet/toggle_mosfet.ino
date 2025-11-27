const int mosfetPin = 9;
bool mosfetState = false;

unsigned long lastToggleTime = 0;
const unsigned long toggleInterval = 1500; // 1 second

void setup() {
  Serial.begin(9600);
  pinMode(mosfetPin, OUTPUT);
  digitalWrite(mosfetPin, LOW);
}

void loop() {
  unsigned long now = millis();

  // Time to toggle?
  if (now - lastToggleTime >= toggleInterval) {
    lastToggleTime = now;


    // Toggle the MOSFET
    mosfetState = !mosfetState;
    digitalWrite(mosfetPin, mosfetState ? HIGH : LOW);
    Serial.print("\nMosfet toggled ");Serial.print(mosfetState);
  }
}
