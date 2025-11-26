#include <Wire.h>

void setup() {
  Serial.begin(9600);
  while (!Serial) {}

  Serial.println("I2C Scanner (UNO R4 Minima)");
  Wire.begin();
}

void loop() {
  byte count = 0;

  for (byte address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    if (Wire.endTransmission() == 0) {
      Serial.print("I2C device found at 0x");
      Serial.println(address, HEX);
      count++;
      delay(10);
    }
  }

  if (count == 0) {
    Serial.println("No I2C devices found.");
  }

  delay(2000);
}
