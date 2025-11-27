#include <Wire.h>
#include "Adafruit_TCS34725.h"

const int THRESHOLD_C = 900;
const int mosfetPin = 9;

unsigned long mosfetOnTime = 0;
unsigned long mosfetOffTime = 0;

const unsigned long autoOffDelay = 1000;   // 1 second max ON
const unsigned long cooldownDelay = 1000;  // 1 second cooldown

bool mosfetOn = false;

Adafruit_TCS34725 tcs = Adafruit_TCS34725(
  TCS34725_INTEGRATIONTIME_50MS,
  TCS34725_GAIN_4X
);

void setMosfetState(bool state) {
  static bool lastState = false;

  if (state && !lastState) {
    // Turning ON
    digitalWrite(mosfetPin, HIGH);
    mosfetOnTime = millis();
  }
  else if (!state && lastState) {
    // Turning OFF
    digitalWrite(mosfetPin, LOW);
    mosfetOffTime = millis();
  }

  lastState = state;
}

void setup() {
  Serial.begin(9600);
  pinMode(mosfetPin, OUTPUT);
  digitalWrite(mosfetPin, LOW);

  if (!tcs.begin()) {
    Serial.println("ERROR: TCS34725 not detected!");
    while (1);
  }
}

void loop() {
  uint16_t r, g, b, c;
  tcs.getRawData(&r, &g, &b, &c);

  unsigned long now = millis();

  // --- 1) AUTO-OFF if ON too long ---
  if (mosfetOn && (now - mosfetOnTime >= autoOffDelay)) {
    mosfetOn = false;      // force off
  }

  // --- 2) If OFF but still in cooldown, do nothing ---
  else if (!mosfetOn && (now - mosfetOffTime < cooldownDelay)) {
    // in cooldown: cannot turn back on
  }

  // --- 3) Otherwise, normal trigger logic ---
  else {
    if (c > THRESHOLD_C) {
      mosfetOn = true;     // will record on-time in setMosfetState()
    } else {
      mosfetOn = false;    // starts cooldown
    }
  }

  setMosfetState(mosfetOn);
}
