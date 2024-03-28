ABOUT_APP = """
        <div class="nice-div">
                        </br>
            <h1> Detector Application</h1>
            <h2> About: </h2>
                This application is built with streamlit.
                It allows for three major ways of detection:
                    <ul>
                        <li> Detection by uploading image(s) </li>
                        <li> Detection by uploading video(s) </li>
                        <li> Detection using webcam </li>
                    </ul>
                The streamlit library is used for building the interface of the app.
                There are two major models(pre-trained) present in this project:
                        <ul>
                            <li>YOLOv3</li>
                            <li>YOLOv8</li>
                        </ul>
                        <h2>Files and Folders:</h2>
                There are multiple files and folders in this project:
                <b>FOLDERS</b>
                    <b>functions</b>
                        This folder contains all the necessary individual functions of the app:
                            <ul>
                                <li>webcam</li>
                                    This function contains all the necessary codes for opening and displaying the detected images using webcam.
                                <li>video_upload</li>
                                    This function contains the codes for
                                        <ul>
                                            <li>uploading</li>
                                            <li>saving</li>
                                            <li>and detecting videos only</li>
                                        </ul>
                                <li>image_upload</li>
                                    This function contains the codes to upload and detect images.
                                <li>settings</li>
                                    This file contains the necessary settings needed for each functions written in the functions folder to avoid repetition.
                                <li>helper</li>
                                    This functions contains various functions that aids the main app file outside the functions folder.
                            </ul>
                    <ul>
                        <li><b>images</b></li>
                            This folder contains local images used for testing the app.
                        <li><b>weights</b></li>
                        This folder contains the different YOLO weights used in the project.
                        <li><b>uploaded_videos</b></li>
                        This is a `code` generated folder that will store the uploaded videos to be re-read for detection in the upload_video function.
                    </ul>
                    <b>FILES</b>
                    <ul>
                        <li><b>main</b></li>
                            This is the main file that contains the streamlit interface code and the calling of the various functions in the functions folder.
                    </ul>
                    <i>NB: Test the app automatically here:</i>
        </div>"""

ABOUT_DEV = """
        <div class="nice-div">
                    <br/>
                    I am a passionate and results-driven Data Scientist with a strong background in analyzing complex data sets and deriving valuable insights to drive business growth. With a deep understanding of machine learning algorithms, and data visualization techniques,
                    I thrive on solving intricate problems and uncovering hidden patterns to optimize decision-making processes.
                    <br/>

        <b> Education: </b>
                    <br/>
                    I am an Engineering student at the University of Ibadan. My academic journey has assisted in developing a solid foundation in mathematics, engineering, programming and problem solving skills, which I have further honed through hands-on experience in various real-world projects.

        <b> Professional Experience: </b>
                    <br/>
                    I have more than 3 years of experience as a Data Scientist, working with diverse organizations as an intern. In my previous role at Zummit Africa, I successfully worked on several projects that involved developing predictive models and Natural language processing. My work directly contributed to the success of my team.

        Furthermore, I collaborated closely with cross-functional teams, including data engineers and machine learning engineers, to work on numerous projects including Large Language Models and computer vision. My expertise in programming languages such as Python and SQL, coupled with my proficiency in data manipulation and analysis tools, allowed me to effectively extract, clean, and transform large datasets for analysis.

        <b> Skills: </b>
        <ul>
                <li> command line </li>
                <li> Data analysis and interpretation </li>
                <li> Machine learning algorithms and predictive modeling </li>
                <li> ComputerVision </li>
                <li> Data visualization </li>
                <li> Programming languages: Python, SQL </li>
                <li> Data manipulation and analysis tools: Pandas, NumPy </li>
                <li> Data Visualization tools: Matplotlib, seaborn </li>
                <li> Deep learning framework:PyTorch </li>
                <li> Data extraction and transformation </li>
        </ul>

        If you are seeking a dedicated and experienced Data Scientist who can bring valuable insights to your organization, feel free to reach out. I am eager to explore new opportunities, contribute to impactful projects, and be part of a team that shares a vision for harnessing the power of data to drive success.
            </div>
                """

STYLE = """
        <style>
            .nice-div {
            font-family: 'Pacifico', cursive;
            font-size: 20px;
            color: #003366;
            padding: 10px;
            margin: 20px;
            border: 2px solid #003366;
            border-radius: 10px;
            }
            .links {
                display: flex;
                font-family: 'Pacifico', cursive;
                font-size: 20px;
                color: #003366;

                height: 100vh;
            }
            a {
                margin: 10px;
                color: #000;
                text-decoration: none;
            }
        </style>"""

LINK = """
        <head>
                <!-- Add font awesome icon library -->
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
        </head>
                <div class="links">
                    <b> Follow/Reach out to me on:<b>
                    <ul>
                        <li><a href="https://github.com/Taoheed-O/" target="_blank"><i class="fa fa-github"></i>  Github</a></li>
                        <li><a href="https://www.linkedin.com/in/taoheed-oyeniyi/" target="_blank"><i class="fa fa-linkedin"></i>  LinkedIn</a></li>
                    </ul>
                </div>
                """
