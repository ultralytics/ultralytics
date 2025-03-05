---
comments: true
description: Discover Ultralytics HUB Cloud Training for easy model training. Upgrade to Pro and start training with a single click. Streamline your workflow now!.
keywords: Ultralytics HUB, cloud training, model training, Pro Plan, easy AI setup
---

# Ultralytics HUB Cloud Training

We've listened to the high demand and widespread interest and are thrilled to unveil [Ultralytics HUB](https://www.ultralytics.com/hub) Cloud Training, offering a single-click training experience for our [Pro](./pro.md) users!

[Ultralytics HUB](https://www.ultralytics.com/hub) [Pro](./pro.md) users can finetune [Ultralytics HUB](https://www.ultralytics.com/hub) models on a custom dataset using our Cloud Training solution, making the model training process easy. Say goodbye to complex setups and hello to streamlined workflows with [Ultralytics HUB](https://www.ultralytics.com/hub)'s intuitive interface.

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ie3vLUDNYZo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> New Feature 🌟 Introducing Ultralytics HUB Cloud Training
</p>

## Train Model

In order to train models using Ultralytics Cloud Training, you need to [upgrade](./pro.md#upgrade) to the [Pro Plan](./pro.md).

Follow the [Train Model](./models.md#train-model) instructions from the [Models](./models.md) page until you reach the third step ([Train](./models.md#3-train)) of the **Train Model** dialog. Once you are on this step, simply select the training duration (Epochs or Timed), the training instance, the payment method, and click the **Start Training** button. That's it!

![Ultralytics HUB screenshot of the Train Model dialog with arrows pointing to the Cloud Training options and the Start Training button](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-dialog.avif)

??? note

    When you are on this step, you have the option to close the **Train Model** dialog and start training your model from the Model page later.

    ![Ultralytics HUB screenshot of the Model page with an arrow pointing to the Start Training card](https://github.com/ultralytics/docs/releases/download/0/hub-cloud-training-model-page-start-training.avif)

Most of the time, you will use the Epochs training. The number of epochs can be adjusted on this step (if the training didn't start yet) and represents the number of times your dataset needs to go through the cycle of train, label, and test. The exact pricing based on the number of epochs is hard to determine, reason why we only allow the [Account Balance](./pro.md#account-balance) payment method.

!!! note

    When using the Epochs training, your [account balance](./pro.md#account-balance) needs to be at least US$5.00 to start training. In case you have a low balance, you can top-up directly from this step.

    ![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to the Top-Up button](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-dialog-top-up.avif)

!!! note

    When using the Epochs training, the [account balance](./pro.md#account-balance) is deducted after every [epoch](https://www.ultralytics.com/glossary/epoch).

    Also, after every epoch, we check if you have enough [account balance](./pro.md#account-balance) for the next epoch. In case you don't have enough [account balance](./pro.md#account-balance) for the next epoch, we will stop the training session, allowing you to resume training your model from the last checkpoint saved.

    ![Ultralytics HUB screenshot of the Model page with an arrow pointing to the Resume Training button](https://github.com/ultralytics/docs/releases/download/0/hub-cloud-training-resume-training-button.avif)

Alternatively, you can use the Timed training. This option allows you to set the training duration. In this case, we can determine the exact pricing. You can pay upfront or using your [account balance](./pro.md#account-balance).

If you have enough [account balance](./pro.md#account-balance), you can use the [Account Balance](./pro.md#account-balance) payment method.

![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to the Start Training button](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-start-training.avif)

If you don't have enough [account balance](./pro.md#account-balance), you won't be able to use the [Account Balance](./pro.md#account-balance) payment method. You can pay upfront or top-up directly from this step.

![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to the Pay Now button](https://github.com/ultralytics/docs/releases/download/0/hub-cloud-training-train-model-pay-now-button.avif)

Before the training session starts, the initialization process spins up a dedicated instance equipped with GPU resources, which can sometimes take a while depending on the current demand and availability of GPU resources.

![Ultralytics HUB screenshot of the Model page during the initialization process](https://github.com/ultralytics/docs/releases/download/0/model-page-initialization-process.avif)

!!! note

    The account balance is not deducted during the initialization process (before the training session starts).

After the training session starts, you can monitor each step of the progress.

If needed, you can stop the training by clicking on the **Stop Training** button.

![Ultralytics HUB screenshot of the Model page of a model that is currently training with an arrow pointing to the Stop Training button](https://github.com/ultralytics/docs/releases/download/0/model-page-training-stop-button.avif)

!!! note

    You can resume training your model from the last checkpoint saved.

    ![Ultralytics HUB screenshot of the Model page with an arrow pointing to the Resume Training button](https://github.com/ultralytics/docs/releases/download/0/hub-cloud-training-resume-training-button.avif)

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/H3qL8ImCSV8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Pause and Resume Model Training Using Ultralytics HUB
</p>

!!! note

    Unfortunately, at the moment, you can only train one model at a time using Ultralytics Cloud.

    ![Ultralytics HUB screenshot of the Train Model dialog with the Ultralytics Cloud unavailable](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-train-model-dialog-1.avif)

## Billing

During training or after training, you can check the cost of your model by clicking on the **Billing** tab. Furthermore, you can download the cost report by clicking on the **Download** button.

![Ultralytics HUB screenshot of the Billing tab inside the Model page with an arrow pointing to the Billing tab and one to the Download button](https://github.com/ultralytics/docs/releases/download/0/hub-cloud-training-billing-tab.avif)
