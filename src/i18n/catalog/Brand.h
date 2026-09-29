#pragma once

// 品牌名称作为消息数据管理；拉丁字标用于标志，本地化名称用于窗口标题和正文。
// 中文统一为简繁同形的“旋影工坊”；日语、韩语固定音译并保留 Spirula Studio，避免名称分化。俄语标题保留拉丁品牌名。

#include "i18n/BeginCatalog.h"

namespace spirula {
namespace i18n {
namespace msg {
namespace brand {

SS_MSG(product,
    EN("Spirula Studio"),      JA("スピルラ・スタジオ"),
    ZH_HANS("旋影工坊"),        ZH_HANT("旋影工坊"),
    KO("스피룰라 스튜디오"),      DE("Spirula Studio"),
    FR("Spirula Studio"),      ES("Spirula Studio"),
    PT("Spirula Studio"),      IT("Spirula Studio"),
    NL("Spirula Studio"),      RU("Spirula Studio"),
    TR("Spirula Studio"));

// 窗口标题逐语言完整定义；本地化名称旁保留拉丁品牌，便于与下载文件和任务管理器进程对应，不能由片段拼接。
SS_MSG(window_title,
    EN("Spirula Studio"),
    JA("スピルラ・スタジオ — Spirula Studio"),
    ZH_HANS("旋影工坊 — Spirula Studio"),
    ZH_HANT("旋影工坊 — Spirula Studio"),
    KO("스피룰라 스튜디오 — Spirula Studio"),
    DE("Spirula Studio"),
    FR("Spirula Studio"),
    ES("Spirula Studio"),
    PT("Spirula Studio"),
    IT("Spirula Studio"),
    NL("Spirula Studio"),
    RU("Spirula Studio"),
    TR("Spirula Studio"));

// 首页字标下方及关于窗口中的单行介绍。
SS_MSG(tagline,
    EN("Reconstruct 3D scenes from photos with Gaussian splatting."),
    JA("写真からガウススプラッティングで3Dシーンを再構成します。"),
    ZH_HANS("用高斯泼溅从照片重建三维场景。"),
    ZH_HANT("以高斯潑濺從相片重建三維場景。"),
    KO("사진에서 가우시안 스플래팅으로 3D 장면을 재구성합니다."),
    DE("Rekonstruiert 3D-Szenen aus Fotos mit Gaussian Splatting."),
    FR("Reconstruit des scènes 3D à partir de photos par Gaussian splatting."),
    ES("Reconstruye escenas 3D a partir de fotos con Gaussian splatting."),
    PT("Reconstrói cenas 3D a partir de fotos com Gaussian splatting."),
    IT("Ricostruisce scene 3D da fotografie con il Gaussian splatting."),
    NL("Reconstrueert 3D-scènes uit foto's met Gaussian splatting."),
    RU("Восстанавливает трёхмерные сцены из фотографий методом гауссова сплаттинга."),
    TR("Fotoğraflardan Gaussian splatting ile 3B sahneler oluşturur."));

SS_MSG(about_line,
    EN("Trains 3D Gaussian Splatting models."),
    JA("3Dガウススプラッティングのモデルを学習します。"),
    ZH_HANS("训练三维高斯泼溅模型。"),
    ZH_HANT("訓練三維高斯潑濺模型。"),
    KO("3D 가우시안 스플래팅 모델을 학습합니다."),
    DE("Trainiert 3D-Gaussian-Splatting-Modelle."),
    FR("Entraîne des modèles 3D Gaussian Splatting."),
    ES("Entrena modelos de 3D Gaussian Splatting."),
    PT("Treina modelos de 3D Gaussian Splatting."),
    IT("Addestra modelli 3D Gaussian Splatting."),
    NL("Traint 3D Gaussian Splatting-modellen."),
    RU("Обучает модели 3D Gaussian Splatting."),
    TR("3D Gaussian Splatting modelleri eğitir."));

}  // 命名空间 brand
}  // 命名空间 msg
}  // 命名空间 i18n
}  // 命名空间 spirula

#include "i18n/EndCatalog.h"
