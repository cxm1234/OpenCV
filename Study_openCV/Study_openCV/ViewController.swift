//
//  ViewController.swift
//  Study_openCV
//
//  Created by  generic on 2022/10/11.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var originImage: UIImageView!
    
    @IBOutlet weak var presentImage: UIImageView!
    
    var takingPicture: UIImagePickerController?
    
    let process = CVProcess()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
    }
    
    enum ImageSource {
    case camera
    case aibum
    }

    // 导入图片
    @IBAction func generator(_ sender: Any) {
        
        let actionSheetController = UIAlertController()
        
        let cancelAction = UIAlertAction(
            title: "取消",
            style: .cancel
        ) { _ in
            print("Tap 取消")
        }
        
        let takingPictureAction = UIAlertAction(
            title: "拍照",
            style: .destructive
        ) { _ in
            self.getImage(source: .camera)
        }
        
        let photoAibumAction = UIAlertAction(
            title: "相册",
            style: .default
        ) { _ in
            self.getImage(source: .aibum)
        }
        
        actionSheetController.addAction(cancelAction)
        actionSheetController.addAction(takingPictureAction)
        actionSheetController.addAction(photoAibumAction)
        
        actionSheetController.popoverPresentationController?.sourceView = sender as? UIView
        
        present(actionSheetController, animated: true)
    }
    
    // 产生结果
    @IBAction func getResult(_ sender: Any) {
        
        guard let originImg = originImage.image else {
            return
        }
        process.handleImg(originImg)
        
        let image = process.sketch()
        
        presentImage.image = image
    }
    
    private func getImage(source: ImageSource) {
        
        takingPicture = UIImagePickerController()
        switch source {
        case .camera:
            takingPicture?.sourceType = .camera
        case .aibum:
            takingPicture?.sourceType = .photoLibrary
        }
        takingPicture?.allowsEditing = false
        takingPicture?.delegate = self
        if let takingPicture = takingPicture {
            present(takingPicture, animated: true)
        }
    }
    
}

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        if takingPicture?.allowsEditing == false {
            originImage.image = info[UIImagePickerController.InfoKey.originalImage] as? UIImage
        } else {
            originImage.image = info[UIImagePickerController.InfoKey.editedImage] as? UIImage
        }
        
        picker.dismiss(animated: true)
    }
    
}
