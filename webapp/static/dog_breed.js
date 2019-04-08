$("#btn-upload-image").click(function () {
    $('#upload-image-dialog').modal();
})

$("#btn-try-me").click(function(){
    $('#upload-image-dialog').modal();
})

// Initialize drop file zone
Dropzone.autoDiscover = false;
var myDropzone = new Dropzone("div#upload-area", { 
    url: "/upload_image",
    paramName: "uploaded_img",
    maxFilesize: 60,
    maxFiles: 1,
    acceptedFiles: ".jpg,.jpeg,.png",
    dictDefaultMessage: "Drop file here or click to select",
    autoProcessQueue: false,
    forceFallback: false
});

myDropzone.on("addedfile", function(file) { 
    if (this.files.length > 1) { 
        this.removeFile(this.files[0]) 
    };
});

$("#upload-image-dialog").on("hidden.bs.modal", function() {
    myDropzone.removeAllFiles();
})

$("#upload_button").on("click", function(event) {
    $("#source_image").attr("src", "");
    event.preventDefault();
    myDropzone.processQueue();
})

myDropzone.on("success", function(file, serverResponse){
    location.href="#Classify";
    setTimeout(function(){$("#upload-image-dialog").modal('hide');}, 800);
    classify_image(serverResponse.filename);
})

// create slider

// Classifying GIF div
function inference_gif_div() {
    var div_gif = $("<div/>").addClass("text-center");
    $("<img/>").attr("src", "static/img/Spinner.gif").appendTo(div_gif);
    $("<h4/>").addClass("analyzing_text").html("Clasifying ...").appendTo(div_gif);
    return div_gif;
}


// Run inference
function classify_image(filename) {
    $("#source_image").attr("src", "/uploads/" + filename);
    var right_div = $("#right_col");
    right_div.empty();
    right_div.append(inference_gif_div());

    $.ajax({
        url: "run_inference?filename=" + filename,
        success: function(results){
            setTimeout(function(){
                right_div.empty();
                $("<p/>").addClass("breed_prediction").html(results["message"]).appendTo(right_div);
            }, 100);
        },
        type: "GET"
    });
}


// Preset examples
$("#brittany").on("click", function(event) {
    event.preventDefault();
    classify_image("Brittany_02625.jpg");
    location.href="#Classify";
})

$("#curly_coated").on("click", function(event) {
    event.preventDefault();
    classify_image("Curly-coated_retriever_03896.jpg");
    location.href="#Classify";
})

$("#human5").on("click", function(event) {
    event.preventDefault();
    classify_image("human_5.jpg");
    location.href="#Classify";
})

$("#petsitting").on("click", function(event) {
    event.preventDefault();
    classify_image("petsitting_pawsitive.png");
    location.href="#Classify";
})

$("#labrador").on("click", function(event) {
    event.preventDefault();
    classify_image("Labrador_retriever_06457.jpg");
    location.href="#Classify";
})