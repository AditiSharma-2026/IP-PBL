from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import io
import base64
import time

app = Flask(__name__)


# =========================================================
# IMAGE TO BASE64
# =========================================================

def image_to_base64(img):
    success, buffer = cv2.imencode(".jpg", img)

    if not success:
        return ""

    return base64.b64encode(buffer).decode("utf-8")


def figure_to_base64():
    buffer = io.BytesIO()

    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=120)
    plt.close()

    buffer.seek(0)

    return base64.b64encode(
        buffer.read()
    ).decode("utf-8")


# =========================================================
# FILTERS
# =========================================================

def apply_filter(img, filter_name, kernel):

    if filter_name == "Gaussian":
        return cv2.GaussianBlur(
            img,
            (kernel, kernel),
            0
        )

    elif filter_name == "Median":
        return cv2.medianBlur(
            img,
            kernel
        )

    elif filter_name == "Mean":
        return cv2.blur(
            img,
            (kernel, kernel)
        )

    elif filter_name == "Bilateral":
        return cv2.bilateralFilter(
            img,
            kernel,
            75,
            75
        )

    return img.copy()


# =========================================================
# NOISE
# =========================================================

def add_noise(img, noise_type):

    if noise_type == "None":
        return img.copy()

    # Gaussian Noise
    if noise_type == "Gaussian":

        noise = np.random.normal(
            0,
            18,
            img.shape
        )

        noisy = (
            img.astype(np.float32)
            + noise
        )

        return np.clip(
            noisy,
            0,
            255
        ).astype(np.uint8)

    # Salt and Pepper Noise
    if noise_type == "Salt & Pepper":

        noisy = img.copy()

        probability = 0.02

        salt = (
            np.random.random(img.shape)
            < probability / 2
        )

        pepper = (
            np.random.random(img.shape)
            < probability / 2
        )

        noisy[salt] = 255
        noisy[pepper] = 0

        return noisy

    # Speckle Noise
    if noise_type == "Speckle":

        noise = np.random.randn(
            *img.shape
        )

        noisy = (
            img.astype(np.float32)
            + img.astype(np.float32)
            * noise
            * 0.15
        )

        return np.clip(
            noisy,
            0,
            255
        ).astype(np.uint8)

    return img.copy()


# =========================================================
# ORB FEATURE MATCHING
# =========================================================

def orb_match(img1, img2):

    start_time = time.perf_counter()

    # Create ORB detector
    orb = cv2.ORB_create(
        nfeatures=500
    )

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(
        img1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        img2,
        None
    )

    if des1 is None or des2 is None:

        raise ValueError(
            "ORB could not find enough features "
            "in one of the images."
        )

    # Brute Force Matcher
    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    # Match descriptors
    matches = bf.match(
        des1,
        des2
    )

    # Sort matches by distance
    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    processing_time = (
        time.perf_counter()
        - start_time
    ) * 1000

    total_matches = len(matches)

    # Best 30 matches
    best_matches = matches[:30]

    distances = [
        m.distance
        for m in matches
    ]

    if distances:

        min_distance = min(
            distances
        )

        max_distance = max(
            distances
        )

        avg_distance = float(
            np.mean(distances)
        )

        # Good match threshold
        good_matches = sum(
            d < 50
            for d in distances
        )

        # Matching score proxy
        matching_score = (
            good_matches
            / total_matches
            * 100
            if total_matches > 0
            else 0
        )

    else:

        min_distance = 0
        max_distance = 0
        avg_distance = 0
        good_matches = 0
        matching_score = 0

    # Draw best matches
    result = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        best_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return {

        "image": image_to_base64(
            result
        ),

        "keypoints1": len(kp1),

        "keypoints2": len(kp2),

        "matches": total_matches,

        "best_matches": len(
            best_matches
        ),

        "good_matches": good_matches,

        "matching_score": round(
            matching_score,
            2
        ),

        "min_distance": round(
            min_distance,
            2
        ),

        "max_distance": round(
            max_distance,
            2
        ),

        "avg_distance": round(
            avg_distance,
            2
        ),

        "time": round(
            processing_time,
            2
        )
    }


# =========================================================
# HISTOGRAM
# =========================================================

def create_histogram(img):

    plt.figure(
        figsize=(7, 4)
    )

    plt.hist(
        img.ravel(),
        bins=256,
        range=(0, 256)
    )

    plt.title(
        "Input Image Histogram"
    )

    plt.xlabel(
        "Pixel Intensity"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.grid(
        alpha=0.2
    )

    return figure_to_base64()


# =========================================================
# MATCHING SCORE GRAPH
# =========================================================

def create_match_graph(results):

    names = [
        r["name"]
        for r in results
    ]

    scores = [
        r["matching_score"]
        for r in results
    ]

    plt.figure(
        figsize=(8, 4)
    )

    plt.bar(
        names,
        scores
    )

    plt.title(
        "ORB Matching Score"
    )

    plt.xlabel(
        "Comparison Image"
    )

    plt.ylabel(
        "Matching Score (%)"
    )

    plt.xticks(
        rotation=20
    )

    return figure_to_base64()


# =========================================================
# PROCESSING TIME GRAPH
# =========================================================

def create_time_graph(results):

    names = [
        r["name"]
        for r in results
    ]

    times = [
        r["time"]
        for r in results
    ]

    plt.figure(
        figsize=(8, 4)
    )

    plt.plot(
        names,
        times,
        marker="o"
    )

    plt.title(
        "ORB Processing Time"
    )

    plt.xlabel(
        "Comparison Image"
    )

    plt.ylabel(
        "Time (ms)"
    )

    plt.xticks(
        rotation=20
    )

    plt.grid(
        alpha=0.3
    )

    return figure_to_base64()


# =========================================================
# KERNEL SIZE ANALYSIS
# =========================================================

def kernel_analysis(
    reference,
    comparison
):

    sizes = [3, 5, 7]

    scores = []

    for size in sizes:

        filtered1 = apply_filter(
            reference,
            "Gaussian",
            size
        )

        filtered2 = apply_filter(
            comparison,
            "Gaussian",
            size
        )

        try:

            result = orb_match(
                filtered1,
                filtered2
            )

            scores.append(
                result["matching_score"]
            )

        except ValueError:

            scores.append(0)

    plt.figure(
        figsize=(7, 4)
    )

    plt.plot(
        sizes,
        scores,
        marker="o"
    )

    plt.title(
        "Effect of Kernel Size"
    )

    plt.xlabel(
        "Kernel Size"
    )

    plt.ylabel(
        "Matching Score (%)"
    )

    plt.xticks(
        sizes
    )

    plt.grid(
        alpha=0.3
    )

    return figure_to_base64()


# =========================================================
# NOISE ANALYSIS
# =========================================================

def noise_analysis(
    reference,
    comparison
):

    noise_types = [
        "None",
        "Gaussian",
        "Salt & Pepper",
        "Speckle"
    ]

    scores = []

    for noise in noise_types:

        noisy1 = add_noise(
            reference,
            noise
        )

        noisy2 = add_noise(
            comparison,
            noise
        )

        try:

            result = orb_match(
                noisy1,
                noisy2
            )

            scores.append(
                result["matching_score"]
            )

        except ValueError:

            scores.append(0)

    plt.figure(
        figsize=(8, 4)
    )

    plt.bar(
        noise_types,
        scores
    )

    plt.title(
        "ORB Robustness to Noise"
    )

    plt.xlabel(
        "Noise Type"
    )

    plt.ylabel(
        "Matching Score (%)"
    )

    plt.xticks(
        rotation=15
    )

    return figure_to_base64()


# =========================================================
# WEBPAGE
# =========================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>ORB Feature Matching</title>

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family: Arial, sans-serif;

    background: #FFF9E8;

    color: #40506A;
}


/* HEADER */

header {

    background:
        linear-gradient(
            135deg,
            #BFDDF5,
            #F9E7A7
        );

    padding: 25px;

    text-align: center;

    border-bottom:
        3px solid #9FC5E5;
}

header h1 {

    margin: 0;

    color: #30465F;
}

header p {

    margin-bottom: 0;

    font-size: 16px;
}


/* CONTAINER */

.container {

    width: 92%;

    max-width: 1200px;

    margin: 25px auto;
}


/* CARDS */

.card {

    background: white;

    padding: 22px;

    margin-bottom: 22px;

    border-radius: 15px;

    box-shadow:
        0 3px 12px
        rgba(0,0,0,0.08);
}


/* HEADINGS */

h2 {

    color: #355B7D;

    border-bottom:
        2px solid #BFDDF5;

    padding-bottom: 8px;
}

h3 {

    color: #567A9C;
}


/* GRID */

.grid {

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(220px, 1fr)
        );

    gap: 15px;
}


/* FORM */

label {

    font-weight: bold;

    display: block;

    margin-bottom: 6px;
}

input,
select {

    width: 100%;

    padding: 10px;

    border:
        1px solid #BFDDF5;

    border-radius: 8px;

    background: #FFFDF5;
}


/* BUTTON */

button {

    margin-top: 15px;

    padding: 12px 25px;

    border: none;

    border-radius: 9px;

    background: #A9D1EA;

    color: #30465F;

    font-weight: bold;

    cursor: pointer;
}

button:hover {

    background: #F0D681;
}


/* INFO */

.info {

    background: #FFF5CF;

    padding: 15px;

    border-radius: 10px;

    margin: 10px 0;
}


/* RESULT */

.result {

    border:
        2px solid #BFDDF5;

    padding: 15px;

    border-radius: 12px;

    margin-bottom: 20px;
}

.result img {

    max-width: 100%;

    border-radius: 8px;
}


/* TABLE */

table {

    width: 100%;

    border-collapse: collapse;

    margin-top: 15px;
}

th {

    background: #BFDDF5;
}

th,
td {

    padding: 9px;

    border:
        1px solid #D7E5EF;

    text-align: center;
}


/* CHART */

.chart {

    width: 100%;

    max-width: 800px;

    display: block;

    margin: 15px auto;

    border-radius: 10px;
}


/* FOOTER */

footer {

    text-align: center;

    padding: 20px;

    background: #BFDDF5;

    margin-top: 30px;
}


/* LOADING */

#loading {

    display: none;

    font-weight: bold;

    margin-top: 15px;
}


/* ERROR */

.error {

    background: #FFE0E0;

    padding: 12px;

    border-radius: 8px;

    color: #8B3333;
}

</style>

</head>


<body>


<header>

<h1>
ORB Feature Matching
</h1>

<p>
Feature Matching using ORB Descriptors
</p>

</header>


<div class="container">

<!-- INTRODUCTION SECTION -->
<div class="card">
<h2>Welcome to ORB Feature Analysis</h2>
<p>
This application performs robust local feature detection and matching using the <b>ORB (Oriented FAST and Rotated BRIEF)</b> algorithm. 
Upload a reference image along with 4 target images from your computer to analyze keypoint correspondences, compute Hamming distances, 
evaluate noise resilience, and benchmark matching accuracy interactively.
</p>
</div>


<!-- =====================================================
     EXPERIMENT
====================================================== -->

<div class="card">

<h2>
1. Experiment Setup
</h2>

<form
    id="form"
    enctype="multipart/form-data"
>


<div class="grid">


<div>

<label>
Image 1 - Reference
</label>

<input
    type="file"
    name="images"
    accept="image/*"
    required
>

</div>


<div>

<label>
Image 2
</label>

<input
    type="file"
    name="images"
    accept="image/*"
    required
>

</div>


<div>

<label>
Image 3
</label>

<input
    type="file"
    name="images"
    accept="image/*"
    required
>

</div>


<div>

<label>
Image 4
</label>

<input
    type="file"
    name="images"
    accept="image/*"
    required
>

</div>


<div>

<label>
Image 5
</label>

<input
    type="file"
    name="images"
    accept="image/*"
    required
>

</div>


<div>

<label>
Noise
</label>

<select name="noise">

<option>
None
</option>

<option>
Gaussian
</option>

<option>
Salt & Pepper
</option>

<option>
Speckle
</option>

</select>

</div>


<div>

<label>
Filter
</label>

<select name="filter">

<option>
None
</option>

<option>
Gaussian
</option>

<option>
Median
</option>

<option>
Mean
</option>

<option>
Bilateral
</option>

</select>

</div>


<div>

<label>
Kernel Size
</label>

<select name="kernel">

<option value="3">
3 × 3
</option>

<option value="5">
5 × 5
</option>

<option value="7">
7 × 7
</option>

</select>

</div>


</div>


<button type="submit">
Run ORB Analysis
</button>

</form>


<div id="loading">

Processing images... Please wait.

</div>

</div>


<!-- RESULTS WILL APPEAR HERE -->

<div id="results"></div>


</div>


<footer>

ORB Feature Matching Project |
Computer Vision

</footer>


<!-- =====================================================
     JAVASCRIPT
====================================================== -->

<script>

document
.getElementById("form")
.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();


        const loading =
            document.getElementById(
                "loading"
            );

        const output =
            document.getElementById(
                "results"
            );


        loading.style.display =
            "block";

        output.innerHTML =
            "";


        const formData =
            new FormData(this);


        try {

            const response =
                await fetch(
                    "/process",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (
                !response.ok
                || data.error
            ) {

                throw new Error(
                    data.error
                    || "Processing failed."
                );

            }


            let html = "";


            /* MATCHING RESULTS (DISPLAYED FIRST) */

            data.results.forEach(
                r => {

                html += `

                <div class="card result">

                <h3>

                Reference vs
                ${r.name}

                </h3>


                <img
                    src="data:image/jpeg;base64,
                    ${r.image}"
                >


                <p>

                <b>Total Matches:</b>
                ${r.matches}

                &nbsp; | &nbsp;

                <b>Best Matches:</b>
                ${r.best_matches}

                &nbsp; | &nbsp;

                <b>Good Matches:</b>
                ${r.good_matches}

                &nbsp; | &nbsp;

                <b>Matching Score:</b>
                ${r.matching_score}%

                </p>


                <p>

                <b>Hamming Distance:</b>

                Min:
                ${r.min_distance}

                &nbsp;

                Avg:
                ${r.avg_distance}

                &nbsp;

                Max:
                ${r.max_distance}

                </p>


                <p>

                <b>Processing Time:</b>
                ${r.time} ms

                </p>

                </div>

                `;

            });


            /* PERFORMANCE TABLE */

            html += `

            <div class="card">

            <h2>
            Performance Summary Table
            </h2>

            <table>

            <tr>

            <th>Image</th>

            <th>Keypoints Ref.</th>

            <th>Keypoints Image</th>

            <th>Total Matches</th>

            <th>Good Matches</th>

            <th>Score (%)</th>

            <th>Avg. Hamming</th>

            <th>Time (ms)</th>

            </tr>

            `;


            data.results.forEach(
                r => {

                html += `

                <tr>

                <td>
                ${r.name}
                </td>

                <td>
                ${r.keypoints1}
                </td>

                <td>
                ${r.keypoints2}
                </td>

                <td>
                ${r.matches}
                </td>

                <td>
                ${r.good_matches}
                </td>

                <td>
                ${r.matching_score}
                </td>

                <td>
                ${r.avg_distance}
                </td>

                <td>
                ${r.time}
                </td>

                </tr>

                `;

            });


            html += `

            </table>

            </div>

            `;


            /* HISTOGRAM */

            html += `

            <div class="card">

            <h2>
            Reference Image Histogram
            </h2>

            <img
                class="chart"
                src="data:image/png;base64,
                ${data.histogram}"
            >

            </div>

            `;


            /* GRAPHS */

            html += `

            <div class="card">

            <h2>
            Performance Graphs
            </h2>


            <h3>
            Matching Score
            </h3>

            <img
                class="chart"
                src="data:image/png;base64,
                ${data.match_graph}"
            >


            <h3>
            Processing Time
            </h3>

            <img
                class="chart"
                src="data:image/png;base64,
                ${data.time_graph}"
            >

            </div>


            <div class="card">

            <h2>
            Kernel Size Analysis
            </h2>

            <p>

            Gaussian filtering is evaluated
            using 3×3, 5×5 and 7×7 kernels.

            </p>

            <img
                class="chart"
                src="data:image/png;base64,
                ${data.kernel_graph}"
            >

            </div>


            <div class="card">

            <h2>
            Noise Analysis
            </h2>

            <img
                class="chart"
                src="data:image/png;base64,
                ${data.noise_graph}"
            >

            </div>


            <div class="card">

            <h2>
            Conclusion
            </h2>

            <p>

            ORB provides an efficient method
            for detecting and matching local
            image features.

            The combination of ORB descriptors
            and Hamming-distance matching makes
            the method suitable for efficient
            feature matching.

            The experiment also demonstrates
            the influence of filtering, noise
            and kernel size on matching performance.

            </p>

            </div>

            `;


            output.innerHTML =
                html;

        }


        catch(error) {

            output.innerHTML = `

            <div class="card error">

            <b>Error:</b>

            ${error.message}

            <br><br>

            Please make sure that all five
            uploaded files are valid images
            containing visible features.

            </div>

            `;

        }


        finally {

            loading.style.display =
                "none";

        }

    }

);

</script>


</body>

</html>

"""


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template_string(
        HTML
    )


# =========================================================
# PROCESS IMAGES
# =========================================================

@app.route(
    "/process",
    methods=["POST"]
)
def process():

    try:

        files = request.files.getlist(
            "images"
        )


        # At least five images
        if len(files) < 5:

            return jsonify({
                "error":
                "Please upload at least 5 images."
            }), 400


        noise_type = request.form.get(
            "noise",
            "None"
        )


        filter_type = request.form.get(
            "filter",
            "None"
        )


        kernel = int(
            request.form.get(
                "kernel",
                "3"
            )
        )


        # =================================================
        # READ IMAGES
        # =================================================

        images = []


        for file in files[:5]:

            data = file.read()


            array = np.frombuffer(
                data,
                np.uint8
            )


            img = cv2.imdecode(
                array,
                cv2.IMREAD_GRAYSCALE
            )


            if img is None:

                return jsonify({
                    "error":
                    "Could not read image: "
                    + file.filename
                }), 400


            images.append(
                img
            )


        # =================================================
        # PREPROCESS IMAGES
        # =================================================

        processed_images = []


        for img in images:

            noisy = add_noise(
                img,
                noise_type
            )


            filtered = apply_filter(
                noisy,
                filter_type,
                kernel
            )


            processed_images.append(
                filtered
            )


        # =================================================
        # REFERENCE IMAGE
        # =================================================

        reference = (
            processed_images[0]
        )


        # =================================================
        # MATCH REFERENCE WITH OTHER FOUR
        # =================================================

        results = []


        for i in range(1, 5):

            result = orb_match(
                reference,
                processed_images[i]
            )


            result["name"] = (
                files[i].filename
            )


            results.append(
                result
            )


        # =================================================
        # HISTOGRAM
        # =================================================

        histogram = create_histogram(
            reference
        )


        # =================================================
        # PERFORMANCE GRALHS
        # =================================================

        match_graph = create_match_graph(
            results
        )


        time_graph = create_time_graph(
            results
        )


        # =================================================
        # KERNEL ANALYSIS
        # =================================================

        kernel_graph = kernel_analysis(
            images[0],
            images[1]
        )


        # =================================================
        # NOISE ANALYSIS
        # =================================================

        noise_graph = noise_analysis(
            images[0],
            images[1]
        )


        # =================================================
        # SEND RESULTS TO WEBPAGE
        # =================================================

        return jsonify({

            "results":
                results,

            "histogram":
                histogram,

            "match_graph":
                match_graph,

            "time_graph":
                time_graph,

            "kernel_graph":
                kernel_graph,

            "noise_graph":
                noise_graph

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False
    )