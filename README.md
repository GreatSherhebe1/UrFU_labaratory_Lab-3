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
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity
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
11) Проверил работу обученной модели

### Выводы
Исходя из результатов обучения и самой обученной модели, можно сделать выводы:

- Эффективность обучающихся моделей увеличивается экспоненциально => нет необходимости в большом количестве моделей;
- В среднем, после 100000 опытов, модель находит наиэфективнейший алгоритм для решения задачи, после этого результат изменяется с небольшим отклонением, которое можно считать погрешностью при дальнейшем обучении.

## Задание 2
### Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.


## Задание 3
### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.


## Выводы
