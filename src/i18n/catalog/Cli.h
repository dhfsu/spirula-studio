#pragma once

#include "i18n/BeginCatalog.h"

namespace spirula { namespace i18n { namespace msg { namespace cli {

SS_MSG(tool_sfm,
    EN("structure from motion: photos or frames -> cameras"),
    JA("Structure from Motion: 写真やフレームからカメラを求める"),
    ZH_HANS("运动恢复结构：由照片或视频帧求相机"),
    ZH_HANT("運動恢復結構：由照片或影格求相機"),
    KO("Structure from Motion: 사진이나 프레임에서 카메라 구하기"),
    DE("Structure from Motion: Fotos oder Einzelbilder -> Kameras"),
    FR("structure from motion : photos ou images -> caméras"),
    ES("structure from motion: fotos o fotogramas -> cámaras"),
    PT("structure from motion: fotos ou quadros -> câmeras"),
    IT("structure from motion: foto o fotogrammi -> camere"),
    NL("structure from motion: foto's of beelden -> camera's"),
    RU("structure from motion: фотографии или кадры -> камеры"),
    TR("structure from motion: fotoğraf ya da karelerden kameralar"));

SS_MSG(usage_commands,
    EN("Commands:"),
    JA("コマンド:"),
    ZH_HANS("命令："),
    ZH_HANT("命令："),
    KO("명령:"),
    DE("Befehle:"),
    FR("Commandes :"),
    ES("Comandos:"),
    PT("Comandos:"),
    IT("Comandi:"),
    NL("Opdrachten:"),
    RU("Команды:"),
    TR("Komutlar:"));

SS_MSG(usage_per_command_help,
    EN("`spirula <command> --help` describes one of them."),
    JA("それぞれの詳細は `spirula <command> --help` で見られます。"),
    ZH_HANS("`spirula <command> --help` 会说明其中某一个命令。"),
    ZH_HANT("`spirula <command> --help` 會說明其中某一個命令。"),
    KO("각 명령의 설명은 `spirula <command> --help`로 볼 수 있습니다."),
    DE("`spirula <command> --help` beschreibt einen davon."),
    FR("`spirula <command> --help` décrit l'une d'entre elles."),
    ES("`spirula <command> --help` describe cada uno de ellos."),
    PT("`spirula <command> --help` descreve cada um deles."),
    IT("`spirula <command> --help` descrive uno di questi."),
    NL("`spirula <command> --help` beschrijft er één."),
    RU("`spirula <command> --help` описывает любую из них."),
    TR("`spirula <command> --help` bunlardan birini anlatır."));

SS_MSG(usage_lang,
    EN("Every command also takes --lang <code>, which sets the interface "
       "language (or SS_LANG in the environment):"),
    JA("どのコマンドでも --lang <code> を指定でき、表示言語が変わります"
       "（環境変数 SS_LANG でも同じです）:"),
    ZH_HANS("每个命令都接受 --lang <code>，用来设置界面语言（也可以用环境变量 "
            "SS_LANG）："),
    ZH_HANT("每個命令都接受 --lang <code>，用來設定介面語言（也可以用環境變數 "
            "SS_LANG）："),
    KO("모든 명령은 --lang <code>도 받습니다. 인터페이스 언어를 정합니다"
       "(환경 변수 SS_LANG도 같습니다):"),
    DE("Jeder Befehl nimmt außerdem --lang <code> entgegen und setzt damit die "
       "Sprache der Oberfläche (oder SS_LANG in der Umgebung):"),
    FR("Chaque commande accepte aussi --lang <code>, qui règle la langue de "
       "l'interface (ou SS_LANG dans l'environnement) :"),
    ES("Todos los comandos aceptan además --lang <code>, que fija el idioma de "
       "la interfaz (o SS_LANG en el entorno):"),
    PT("Todo comando também aceita --lang <code>, que define o idioma da "
       "interface (ou SS_LANG no ambiente):"),
    IT("Ogni comando accetta anche --lang <code>, che imposta la lingua "
       "dell'interfaccia (oppure SS_LANG nell'ambiente):"),
    NL("Elke opdracht neemt ook --lang <code>, waarmee je de taal van de "
       "interface instelt (of SS_LANG in de omgeving):"),
    RU("Любая команда принимает также --lang <code>, задающий язык интерфейса "
       "(или переменную окружения SS_LANG):"),
    TR("Her komut ayrıca arayüz dilini belirleyen --lang <code> seçeneğini de "
       "alır (ya da ortamdaki SS_LANG):"));

SS_MSG(err_unknown_command,
    EN("error: unknown command '{0}'"),
    JA("エラー: 不明なコマンド '{0}'"),
    ZH_HANS("错误：未知命令 '{0}'"),
    ZH_HANT("錯誤：未知命令 '{0}'"),
    KO("오류: 알 수 없는 명령 '{0}'"),
    DE("Fehler: unbekannter Befehl '{0}'"),
    FR("erreur : commande inconnue '{0}'"),
    ES("error: comando desconocido '{0}'"),
    PT("erro: comando desconhecido '{0}'"),
    IT("errore: comando sconosciuto '{0}'"),
    NL("fout: onbekende opdracht '{0}'"),
    RU("ошибка: неизвестная команда '{0}'"),
    TR("hata: bilinmeyen komut '{0}'"));

SS_MSG(err_no_gui,
    EN("error: this build has no graphical application (-DSS_BUILD_GUI=OFF)"),
    JA("エラー: このビルドにはグラフィカルなアプリケーションが含まれていません"
       "（-DSS_BUILD_GUI=OFF）"),
    ZH_HANS("错误：这个版本没有图形界面应用（-DSS_BUILD_GUI=OFF）"),
    ZH_HANT("錯誤：這個版本沒有圖形介面應用程式（-DSS_BUILD_GUI=OFF）"),
    KO("오류: 이 빌드에는 그래픽 응용 프로그램이 없습니다(-DSS_BUILD_GUI=OFF)"),
    DE("Fehler: dieser Build enthält keine grafische Anwendung "
       "(-DSS_BUILD_GUI=OFF)"),
    FR("erreur : cette compilation n'a pas d'application graphique "
       "(-DSS_BUILD_GUI=OFF)"),
    ES("error: esta compilación no tiene aplicación gráfica "
       "(-DSS_BUILD_GUI=OFF)"),
    PT("erro: esta compilação não tem aplicativo gráfico (-DSS_BUILD_GUI=OFF)"),
    IT("errore: questa build non ha un'applicazione grafica "
       "(-DSS_BUILD_GUI=OFF)"),
    NL("fout: deze build heeft geen grafische toepassing (-DSS_BUILD_GUI=OFF)"),
    RU("ошибка: в этой сборке нет графического приложения "
       "(-DSS_BUILD_GUI=OFF)"),
    TR("hata: bu derlemede grafik arayüzlü uygulama yok (-DSS_BUILD_GUI=OFF)"));

SS_MSG(error_line,
    EN("error: {0}"),
    JA("エラー: {0}"),
    ZH_HANS("错误：{0}"),
    ZH_HANT("錯誤：{0}"),
    KO("오류: {0}"),
    DE("Fehler: {0}"),
    FR("erreur : {0}"),
    ES("error: {0}"),
    PT("erro: {0}"),
    IT("errore: {0}"),
    NL("fout: {0}"),
    RU("ошибка: {0}"),
    TR("hata: {0}"));

SS_MSG(usage_try_help,
    EN("Try '{0}' for more information."),
    JA("詳しくは '{0}' を実行してください。"),
    ZH_HANS("更多信息请运行 '{0}'。"),
    ZH_HANT("更多資訊請執行 '{0}'。"),
    KO("자세한 내용은 '{0}' 을(를) 실행하세요."),
    DE("Mehr dazu mit '{0}'."),
    FR("Pour en savoir plus : '{0}'."),
    ES("Para saber más, ejecuta '{0}'."),
    PT("Para saber mais, execute '{0}'."),
    IT("Per saperne di più: '{0}'."),
    NL("Meer weten? Voer '{0}' uit."),
    RU("Подробнее: '{0}'."),
    TR("Ayrıntı için '{0}' çalıştırın."));

SS_MSG(sfm_merge_output_is_input,
    EN("--output {0} is where the input models live; pass --in-place if that is "
       "what you want"),
    JA("--output {0} は入力モデルのある場所です。意図どおりなら --in-place を"
       "渡してください"),
    ZH_HANS("--output {0} 正是输入模型所在之处；若确实要如此，请加 --in-place"),
    ZH_HANT("--output {0} 正是輸入模型所在之處；若確實要如此，請加 --in-place"),
    KO("--output {0} 은(는) 입력 모델이 있는 곳입니다. 그것이 뜻한 바라면 "
       "--in-place 를 주세요"),
    DE("--output {0} ist der Ort, an dem die Eingabemodelle liegen; übergeben "
       "Sie --in-place, wenn das gewollt ist"),
    FR("--output {0} est là où se trouvent les modèles d'entrée ; passez "
       "--in-place si c'est bien ce que vous voulez"),
    ES("--output {0} es donde están los modelos de entrada; pasa --in-place si "
       "es eso lo que quieres"),
    PT("--output {0} é onde estão os modelos de entrada; passe --in-place se é "
       "isso que você quer"),
    IT("--output {0} è dove stanno i modelli di ingresso; passi --in-place se è "
       "questo che vuole"),
    NL("--output {0} is waar de invoermodellen staan; geef --in-place op als "
       "dat de bedoeling is"),
    RU("--output {0} -- это то место, где лежат входные модели; передайте "
       "--in-place, если так и задумано"),
    TR("--output {0} zaten girdi modellerinin bulunduğu yer; istediğiniz buysa "
       "--in-place verin"));

} } } }

#include "i18n/EndCatalog.h"
