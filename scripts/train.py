import torch
import mlflow
import os
from scripts.evaluate import evaluate_map50
from config.config import load_config


def train_one_epoch(model, optimizer, train_loader, device, epoch):
    """
    Training for one epoch
    """
    model.train()
    total_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_objectness = 0
    loss_rpn_box_reg = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        # Move images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        loss_classifier += loss_dict['loss_classifier'].item()
        loss_box_reg += loss_dict['loss_box_reg'].item()
        loss_objectness += loss_dict['loss_objectness'].item()
        loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()

        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                  f'loss_classifier: {loss_classifier/(batch_idx+1):.4f}, '
                  f'loss_box_reg: {loss_box_reg/(batch_idx+1):.4f}, '
                  f'loss_objectness: {loss_objectness/(batch_idx+1):.4f}, '
                  f'loss_rpn_box_reg: {loss_rpn_box_reg/(batch_idx+1):.4f}, '
                  f'Total Loss: {losses/(batch_idx+1):.4f}')

    all_losses = {
        'loss_classifier': loss_classifier / len(train_loader),
        'loss_box_reg': loss_box_reg / len(train_loader),
        'loss_objectness': loss_objectness / len(train_loader),
        'loss_rpn_box_reg': loss_rpn_box_reg / len(train_loader),
        'Total Loss': total_loss / len(train_loader)
    }

    return all_losses

def train_model(model, train_loader, val_loader, config, device):
    """
    Full training loop
    """
    # Setup optimizer

    print('Training model...')
    params = [p for p in model.parameters() if p.requires_grad]

    # Optimizer
    if config.get('optimizer') == 'adam':
        optimizer = torch.optim.Adam(params, lr=config['learning_rate'], weight_decay=0.0005)
    else:
        optimizer = torch.optim.SGD(params, lr=config['learning_rate'], momentum=0.9, weight_decay=0.0005)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Track best model
    best_map_50 = 0
    best_model_path = os.path.join(config['models_dir'], 'best_model.pth')

    for epoch in range(config['epochs']):
        # Training
        print(f'Epoch: {epoch}')
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Validation
        val_metrics = evaluate_map50(model, val_loader, device)
        print(val_metrics)

        # Update learning rate
        # lr_scheduler.step()

        mlflow.log_metrics({
            'map_50': val_metrics,
            'Total Loss': train_loss['Total Loss'],
            'Learning Rate': optimizer.param_groups[0]['lr'],
            'loss_classifier': train_loss['loss_classifier'],
            'loss_box_reg': train_loss['loss_box_reg'],
            'loss_objectness': train_loss['loss_objectness'],
            'loss_rpn_box_reg': train_loss['loss_rpn_box_reg'],
        }, step=epoch)

        # Save best model
        if val_metrics > best_map_50:
            best_map_50 = val_metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_map_50': best_map_50,
            }, best_model_path)