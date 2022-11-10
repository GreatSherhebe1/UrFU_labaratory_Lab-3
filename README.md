# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил(а):
- Устинов Никита Валерьевич
- РИ-210910
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | # | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения.
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения.
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения.
- Выводы.
- ✨Magic ✨

## Цель работы
познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity. 
## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity

Ход работы:

1) Создал новый пустой 3D проект на Unity;
2) Скачал папку с MLAgent и другие файлы для лабораторной работы;
3) В созданном проекте добавил ML Agent через 2 json файла;
4) Запустил Anaconda Promt от имени администратора и ввел серию команд для скачивания необходимых библиотек
```
conda create -n MlAgent python=3.6.13
conda activate MLAgent

pip install mlagents==0.28.0
pip install torch~=1.7.1 -f https://download.pytorch.org/whl/torch_stable.html
```
![Image alt](https://github.com/GreatSherhebe1/UrFU_labaratory_Lab-3/raw/main/скриншоты/laba3_1.png)
5) Создал на сцене плато(**Floor**), куб(**Target**), сферу (**RollerAgent**);
6) Добавил в скрипт-файл на RollerAgent;

```cs
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    private Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;

    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }

    public float forceMultiplier = 10;

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if (distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```

7) Добавил объекту **RollerAgent** компоненты **Rigidbody**, **Decision Requester**, **Behavior Parameters**, провел настройку из методички;
8) Добавил файл конфигурации нейронной сети в папку проекта;
9) запустил работу mlagent командой;

```
mlagents-learn rollerball_config.yaml -- run-id=RollerBall --force
```

10) Сделал 3, 9, 27 копий модели, пронаблюдал за обучением;
![Image alt](https://github.com/GreatSherhebe1/UrFU_labaratory_Lab-3/raw/main/скриншоты/laba3_2.png)
![Image alt](https://github.com/GreatSherhebe1/UrFU_labaratory_Lab-3/raw/main/скриншоты/laba3_3.png)
![Image alt](https://github.com/GreatSherhebe1/UrFU_labaratory_Lab-3/raw/main/скриншоты/laba3_4.png)
11) Проверил работу обученной модели

### Выводы
Исходя из результатов обучения и самой обученной модели, можно сделать выводы:

- Эффективность обучающихся моделей увеличивается экспоненциально => нет необходимости в большом количестве моделей;
- В среднем, после 100000 опытов, модель находит наиэфективнейший алгоритм для решения задачи, после этого результат изменяется с небольшим отклонением, которое можно считать погрешностью при дальнейшем обучении.

## Задание 2
### Подробно опишите каждую строку файла конфигурации нейронной сети. Самостоятельно найдите информацию о компонентах Decision Requester, Behavior Parameters, добавленных на сфере.

Ход работы:

**1)**
-  Название модели
```
  RollerBall:
```

- Алгоритм обучения модели
```
  trainer_type: ppo
```

- Параметры для обучения модели
```
  hyperparameters:
```

- Количество опыта за итерацию
```
  batch_size: 10
```

- Количество опыта, необходимое для обучения модели, которое должно быть больше чем "batch_size"
```
  buffer_size: 100
```

- Начальная скорость обучения для градиентного спуска. Соответствует силе шага градиентного спуска. При нестабильном обучении следует уменьшить
```
  learning_rate: 3.0e-4
```

- Сила энтропийной регуляризации, которая делает политику случайной, что гарантирует, что агенты правильно исследуют пространство для действия во время эксперимента. Увеличение параметра увеличивает случайность действий. Измеряется с помощью "TensorBoard". Нужно, чтобы медленно уменьшалась с увелечением награды "Reward". следует колибровать обратно пропорционально падению энтропии
```
  beta: 5.0e-4
```

- Переменная влияет на быстроту развития в обучении. Соответствует допустимому порогу между предыдущим и новым опытом при обновлении градиентного спуска. Меньшее значение увеличивает точность, но замедляет работу
```
  epsilon: 0.2
```

- Лямда используется при  расчете обобщенной оценки преимущества (GAE). Можно представить как достоверность для агента текущей оценки стоимости. Обновляет значение стоимости после опыта. Низкое значение увеличивает надежность для агента, а высокое - больше пологаться  на вознаграждения, полученные в среде (может быть высокой дисперсией). Правильное значение стабилизирует обучение.
```
  lambd: 0.99
```

- Количество проходов через буфер опыта при оптимизации градиентного спуска. Уменьшение обеспечивает стабильное обновление за счет медленного обучения
```
  num_epoch: 3
```

- Определяет изменение скорости обучения со временем. linear линейно уменьшает learning_rate, достигая 0 при max_steps
```
  learning_rate_schedule: linear
```

- Настройки нейронной сети
```
  network_settings:
```

- Применяется ли нормализация к входным данным векторного наблюдения. Эта нормализация основана на скользящем среднем и дисперсии векторного наблюдения. Может быть полезна в сложных задачах непрерывного управления, но вредна в простых задачах жискретного управления
```
  normalize: false
```

- Количество блоков в скрытых слоях нейронной сети. Соответствует количеству узлов в каждом полносвязном слое нейронной сети. Для простых задач с простой комбинацией входных данных наблюдения должно быть небольшим. В сложных задачах с тесным взаимодействием между переменными наблюдения должно быть большим
```
  hidden_units: 128
```

- Количество  скрытых слоев в нейронной сети. Соответсвует количеству скрытых слоев после ввода наблюдения или после кодирования визуального наблюдения. Для простых задач- меньше слоев для быстроты и эффективности. В  сложных случаях может понадобиться управление большего числа слоев
```
  num_layers: 2
```

- Действия при вознаграждении
```
  reward_signals:
    extrinsic:
```

- Коэффициент дисконтирования для будущих вознаграждений из окружающей среды. Можно рассматривать как озабоченность агентом вознаграждением из среды в ситуациях, когда агент должен действовать в настоящем, чтобы подготовиться к вознаграждению в дальнейшем, должно быть большим. В случаях, когда вознаграждение быстрое, может быть меньше. Строго меньше 1
```
  gamma: 0.99
```

- Фактор умножения вознаграждения из среды. Диапазоны могут варьироваться в зависимости от сигнала вознаграждения.
```
  strength: 1.0
```

- Общее число шагов собранных наблюдений и предпринятых действий, которое необходимо выполнить для завершения процесса обучения. При нескольких одноименных агентов, будет учитываться одно и то же значение max_steps
```
  max_steps: 500000
```

- Количество опыта для каждого агента, для добавления в буфер опыта. Когда этот предел достигается до конца эпизода, оценка используется для прогнозирования общего ожидаемого вознаграждения из текущего состояния агента. Этот параметр является компромиссом между менее предвзятой и более высокой оценки дисперсии. В тех случаях, когда в эпизоде есть частые награды или эпизоды непомерно велеки, идеальным может быть меньшее количество. Это число должно быть достаточно большим, чтобы охватить все возможные действия в последовательности действий агента
```
  time_horizon: 64
```

- Количество опытов, которое необходимо собрать перед созданием и отображением статистики обучения, определяет детализацию графиков в Tensorboard
```
  summary_freq: 10000
```

**2)**

**Decision Requester** автоматически запрашивает решения для экземпляра агента через регулярные промежутки времени. Прикрепляется к тому же **GameObject**, что и компонент **Agent**. Компонент DecisionRequester предоставляет удобный и гибкий способ запуска процесса принятия решения агентом. Без DecisionRequester реализация агента должна вручную вызывать функцию RequestDecision()

**Behavior Parameters** Компонент для настройки поведения экземпляра агента и свойств brain. Во время выполнения этот компонент генерирует объекты политики агента в соответствии с настройками, указанными в редакторе

## Задание 3
### Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и в первом задании, случайно изменять координаты на плоскости.

Ход работы:

Добавил на сцену новый куб желтого цвета и сделал его целью для **RollerBallAgent**. Добавил в скрипт очередность достижения цели и новый куб.

```cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    private Rigidbody rBody;
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
        Target = FirstTarget;
    }

    public Transform FirstTarget;
    public Transform SecondTarget;
    private Transform Target;

    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        if (Target == FirstTarget)
            Target = SecondTarget;
        else
        {
            Target = FirstTarget;
            FirstTarget.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
            SecondTarget.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }

    public float forceMultiplier = 10;

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if (distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```
![Image alt](https://github.com/GreatSherhebe1/UrFU_labaratory_Lab-3/raw/main/скриншоты/laba3_5.png)
## Выводы
В ходе работы научился работать с некоторыми функциями MLAgent, проводить обучение модели с помощью ML Agent.
Игровой баланс это система весов, которые увеличивают вовлеченность пользователя за счет постоянного изменения состояния экономики в системе, не дающей заскучать или бросить из-за сложности игру. Игровой баланс не идеален, но благодаря инструментам, реализующим ML он становится наиболее реалистичным, подходящим для пользователя по его возможностям.
