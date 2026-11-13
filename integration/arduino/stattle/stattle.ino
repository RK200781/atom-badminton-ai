/*///////////////////////////////////////////////////////////////////////
シャトル発射装置
2026/10/22 ver1

単独動作の場合  
①シャトルを入れる
②SW1を押してローラーを回転させる、押すごとに弱、中、強に切り替わる
③SW2を押してシャトルを発射
注意①シャトルが入っていないとSW1・SW2を押しても反応しない。（初回は動作してしまうバグあり）
注意①シャトルが入っていない状態で10秒経つとモータが自動停止する。

シリアル通信の場合
①シャトルを入れる
②電源を入れる
③115200bpsで接続
④「0,1\n」や「0,2\n」のようにテキストを送信するとローラーが回転し始める(シャトルが入っていないと回転しない)。
一番目の数字はシャトルの発射に関する数値
0:なし、1:発射
二番目の数値はシャトルの強さに関する数値
0:停止、1:弱(10)、2:中(50)、3:強(100)、3～100:それに応じた強さ
⑤「1\n」でシャトルを発射
⑥シャトルが入っていないときは0が送信される。
注意①シャトルが入っていないとローラが回転しないが、入れた瞬間にローラが回転し始めて発射されるので注意
/////////////////////////////////////////////////////////////////////*/




int A[4] = { 0 };

#define IR 2      //赤外線センサ：シャトルがあるか検知する
#define SHATLL 1  //ポテンショメータ：シャトルを補充するやつの角度
#define POT 3     //基板にあるポテンショメータ：なんかの設定用

#define SW1 3  //基板にあるスイッチ：手動モータON、OFF
#define SW2 2  //基板にあるスイッチ：手動装填

#define push_motor 5
#define go_motor 6

// ポテンショメータの実測値に変更
const int POT_MIN = 0;
const int POT_MAX = 1023;

const int SPEED_FAST = 80;
const int SPEED_SLOW = 30;

const int SLOW_POS = 500;
const int STOP_POS = 700;

const unsigned long TIMEOUT = 10000;

bool running = false;
bool armed = true;

int pote_flag = 0;
int prevValue = 0;
int slowValue = 0;
int stopValue = 0;
unsigned long startTime = 0;

const int RANGE = POT_MAX - POT_MIN;

int motor_mode = 0;   // 0=停止、1=弱、2=中、3=強
bool sw1_old = HIGH;  // 前回のSW1の状態

bool shattle;  //シャトルの検知
unsigned long shatll_startTime = 0;

int moniter[10];
int logs[11]={0};

char S_Command;
float getA[10];  //シリアルデータ受信用
bool shot_flag = 0;

void setup() {
  // put your setup code here, to run once:
  pinMode(2, INPUT);
  pinMode(3, INPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(7, OUTPUT);
  digitalWrite(4, 0);
  digitalWrite(5, 0);
  digitalWrite(6, 0);
  digitalWrite(7, 0);
  S_Command = 'X';
  Serial.begin(115200);
}

void loop() {
  Serialgetdata();

  shot_flag = getA[0];  //発射の合図　1だったら発射
  getA[0] = 0;
  motor_mode = getA[1];  //発射の強さ

  for (int i = 0; i < 4; i++) {
    A[i] = analogRead(i);
  }

  shatll_ON();          //シャトルの検知
  shatll_set();         //シャトルの装填
  shot_motor_switch();  //シャトルの打つ強さ

  Serial.println(logs[10]);
  delay(10);
}

void shatll_ON() {
  if (A[IR] < 500) {
    shattle = 1;
    shatll_startTime = millis();
  } else {
    shattle = 0;
    // 安全停止
    if (millis() - shatll_startTime > TIMEOUT) {
      motor_mode = 0;
    }
  }
  logs[9]=shattle;
  int mean;
  mean=logs[0];
  for(int i=1;i<10;i++){
      mean+=logs[i];
      logs[i-1]=logs[i];
  }
  if(mean>5){
    logs[10]=1;
  }else{
    logs[10]=0;
  }
}

void shatll_set() {
  // SW2が押されたらスタート
  if (!running) {
    if (motor_mode != 0 && logs[10] == 1 && digitalRead(SW2) == HIGH) {
      armed = true;
    }
    if (armed && digitalRead(SW2) == LOW) {
      delay(25);

      if (digitalRead(SW2) == LOW) {
        armed = false;
        prevValue = 1023 - analogRead(SHATLL);
        pote_flag = 0;
        startTime = millis();

        running = true;
        shatll_set_motor(SPEED_FAST);
      }
    }

    if (motor_mode != 0 && logs[10] == 1 && shot_flag == LOW) {
      armed = true;
    }
    if (armed && shot_flag == HIGH) {
      delay(25);

      if (shot_flag == HIGH) {
        armed = false;
        prevValue = 1023 - analogRead(SHATLL);
        pote_flag = 0;
        startTime = millis();

        running = true;
        shatll_set_motor(SPEED_FAST);
      }
    }
  }

  if (running) {
    int nowValue = 1023 - analogRead(SHATLL);
    if (pote_flag == 0 && prevValue - nowValue > 200) {
      pote_flag = 1;
    }
    if (pote_flag != 0) {
      if (pote_flag == 1) {
        // 停止手前で減速
        if (nowValue >= SLOW_POS) {
          shatll_set_motor(SPEED_SLOW);
        }
      }
      // 1回転で停止
      if (nowValue >= STOP_POS) {
        shatll_set_motor_stop();
        pote_flag = 2;
        running = false;
        return;
      }
    }

    // 安全停止
    if (millis() - startTime > TIMEOUT) {
      shatll_set_motor_stop();
      running = false;
      //Serial.println("TIMEOUT");
      return;
    }
    delay(5);
  }
}

void shatll_set_motor(int SPEED) {
  analogWrite(push_motor, SPEED);
}

void shatll_set_motor_stop() {
  analogWrite(push_motor, 0);
}

void shot_motor_switch() {
  // SW1を押した瞬間だけ切り替える
  if (digitalRead(3) == LOW && sw1_old == HIGH) {

    motor_mode++;

    if (motor_mode > 3) {
      motor_mode = 0;
    }

    delay(30);  // チャタリング防止
  }

  sw1_old = digitalRead(3);

  // モータ速度の切り替え
  if (motor_mode == 1) {
    set_shot_motor_power(20);  // 弱
  } else if (motor_mode == 2) {
    set_shot_motor_power(50);  // 中
  } else if (motor_mode == 3) {
    set_shot_motor_power(100);  // 強
  } else if (motor_mode > 3 && motor_mode <= 100) {
    set_shot_motor_power(motor_mode);
  } else {
    analogWrite(go_motor, 0);  // 停止
  }
}

void set_shot_motor_power(int power) {
  int PWM = (int)(2.55 * power);
  analogWrite(go_motor, PWM);
}

String Serial_d;
int no = 0;
int flag = 0;

void Serialgetdata() {
  while (Serial.available() > 0) {
    char d;
    String a;
    String b = "\n";
    String c = ",";
    String a0 = "D";
    String a1 = "E";
    String a2 = "C";
    String a3 = "A";
    String a4 = "X";
    int falsedata;
    d = Serial.read();
    a = d;

    if (a.equals(c) == true) {  //'\t'を読み込んだ時
      falsedata = getA[no];
      getA[no] = Serial_d.toFloat();  //文字列を数値化
      Serial_d = "";                  //文字列のクリア
      no++;
    } else if (a.equals(b) == true) {  //改行を読み込んだ時
      falsedata = getA[no];
      getA[no] = Serial_d.toFloat();  //文字列を数値化
      Serial_d = "";                  //文字列のクリア
      no = 0;
    } else {
      if (a.equals(a0) == true || a.equals(a1) == true || a.equals(a2) == true || a.equals(a3) == true || a.equals(a4) == true) {
        S_Command = d;
      } else {
        Serial_d += a;  //取得した文字を文字列に
      }
    }
  }
}
