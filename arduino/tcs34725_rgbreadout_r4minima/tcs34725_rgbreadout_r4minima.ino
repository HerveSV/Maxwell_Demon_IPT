#include <Wire.h>
#include "Adafruit_TCS34725.h"

// Default integration time and gain are fine for raw readings
Adafruit_TCS34725 tcs = Adafruit_TCS34725(
  TCS34725_INTEGRATIONTIME_50MS,
  TCS34725_GAIN_4X
);

// If your sensor’s LED pin is tied to a digital pin, set that pin here.
// If the LED pin is wired directly to 3.3V/5V, set to -1.
const int SENSOR_LED_PIN = -1;  // change if needed

void setup() {
  Serial.begin(9600);
  while (!Serial) { ; }  // Required on UNO R4 Minima
  delay(200);

  if (SENSOR_LED_PIN >= 0) {
    pinMode(SENSOR_LED_PIN, OUTPUT);
    digitalWrite(SENSOR_LED_PIN, HIGH);
  }

  Serial.println("TCS3472 Raw RGBC Reader (UNO R4 Minima)");
  Serial.println("------------------------------------------------");

  if (!tcs.begin()) {
    Serial.println("ERROR: TCS3472 not detected! Check wiring (SDA=20, SCL=21).");
    while (1) { delay(500); }
  }
}

void loop() {
  uint16_t r, g, b, c;

  // Get raw values directly from the sensor
  tcs.getRawData(&r, &g, &b, &c);

  Serial.print("R: "); Serial.print(r);
  Serial.print("  G: "); Serial.print(g);
  Serial.print("  B: "); Serial.print(b);
  Serial.print("  C: "); Serial.println(c);

  delay(250);  // Slow down output for readability
}
