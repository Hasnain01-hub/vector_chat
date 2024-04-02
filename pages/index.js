import Head from "next/head";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/router";
import { logoutSuccess } from "../reducer";
import React from "react";
import { useSelector } from "react-redux";
import { auth } from "../utils/Firebase";
import Header from "../components/Header";
import { useDispatch } from "react-redux";
import axios from "axios";
export default function Home() {
  const route = useRouter();

  const { user } = useSelector((state) => state.user);
  const [navbar, setnavBar] = React.useState(false);
  const dispatch = useDispatch();

  const handleLogout = async (e) => {
    e.preventDefault();
    await auth
      .signOut()
      .then(async () => {
        window.localStorage.removeItem("cart");
        dispatch(logoutSuccess({}));
        toast.success("Logout Success");
        window.location.href = "/";
        dispatch({
          type: "REMOVE_FROM_CART",
          payload: [],
        });

        // setInterval(() => {
        //   window.location.reload();
        // }, 1000);
      })
      .catch((error) => {
        console.log(error.message);
      });
    window.location.reload();
  };
  return (
    <>
      <div>
        <div className="techwave_fn_fixedsub">
          <ul></ul>
        </div>

        {/* <div className='techwave_fn_preloader enabled'>
        <svg>
          <circle className='first_circle' cx='50%' cy='50%' r='110'></circle>
          <circle className='second_circle' cx='50%' cy='50%' r='110'></circle>
        </svg>
      </div> */}

        <div className="techwave_fn_wrapper">
          <div className="techwave_fn_wrap">
            <div className="techwave_fn_searchbar">
              <div className="search__bar">
                <input
                  className="search__input"
                  type="text"
                  placeholder="Search here..."
                />
                <img
                  src="../svg/search.svg"
                  alt=""
                  className="fn__svg search__icon"
                />
                <Link className="search__closer" href="/">
                  <img src="svg/close.svg" alt="" className="fn__svg" />
                </Link>
              </div>
              <div className="search__results">
                <div className="results__title">Results</div>
                <div className="results__list">
                  <ul>
                    <li>
                      <Link href="/">Artificial Intelligence</Link>
                    </li>
                    <li>
                      <Link href="/">
                        Learn about the impact of AI on the financial industry
                      </Link>
                    </li>
                    <li>
                      <Link href="/">
                        Delve into the realm of AI-driven manufacturing
                      </Link>
                    </li>
                    <li>
                      <Link href="/">
                        Understand the ethical implications surrounding AI
                      </Link>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
            <Header
              user={user}
              handleLogout={handleLogout}
              setnavBar={setnavBar}
              navbar={navbar}
            />
            <div className="techwave_fn_leftpanel">
              <div className="mobile_extra_closer"></div>

              <div className="leftpanel_logo">
                <Link
                  href="/"
                  // onClick={() => {
                  //   route.push("/");
                  // }}
                  className="fn_logo"
                >
                  <span className="full_logo">
                    <img
                      src="images/CCPD_Text.png"
                      alt=""
                      className="desktop_logo"
                    />
                    <img
                      src="images/CCPD_Text.png"
                      alt=""
                      className="retina_logo"
                    />
                  </span>
                  <span className="short_logo">
                    <img
                      src="images/logo-desktop-mini.png"
                      alt=""
                      className="desktop_logo"
                    />
                    <img
                      src="images/logo-retina-mini.png"
                      alt=""
                      className="retina_logo"
                    />
                  </span>
                </Link>
                <Link
                  href="/"
                  className="fn__closer fn__icon_button desktop_closer"
                >
                  <img src="svg/arrow.svg" alt="" className="fn__svg" />
                </Link>
                <Link
                  href="/"
                  className="fn__closer fn__icon_button mobile_closer"
                >
                  <img src="svg/arrow.svg" alt="" className="fn__svg" />
                </Link>
              </div>

              <div className="leftpanel_content">
                <div className="nav_group">
                  <h2 className="group__title">Start Here</h2>
                  <ul className="group__list">
                    <li>
                      <Link
                        href="/"
                        onClick={() => {
                          route.push("/");
                        }}
                        className="fn__tooltip active menu__item"
                        data-position="right"
                        title="Home"
                      >
                        <span className="icon">
                          <img src="svg/home.svg" alt="" className="fn__svg" />
                        </span>
                        <span className="text">Home</span>
                      </Link>
                    </li>
                    {/* <li>
                      <Link
                        href="/"
                        className="fn__tooltip menu__item"
                        data-position="right"
                        title="Community Feed"
                      >
                        <span className="icon">
                          <img
                            src="svg/community.svg"
                            alt=""
                            className="fn__svg"
                          />
                        </span>
                        <span className="text">Community Feed</span>
                      </Link>
                    </li>
                    <li>
                      <Link
                        href="/"
                        className="fn__tooltip menu__item"
                        data-position="right"
                        title="Personal Feed"
                      >
                        <span className="icon">
                          <img
                            src="svg/person.svg"
                            alt=""
                            className="fn__svg"
                          />
                        </span>
                        <span className="text">
                          Personal Feed<span className="count">48</span>
                        </span>
                      </Link>
                    </li>
                    <li>
                      <Link
                        href="/"
                        className="fn__tooltip menu__item"
                        data-position="right"
                        title="Finetuned Models"
                      >
                        <span className="icon">
                          <img src="svg/cube.svg" alt="" className="fn__svg" />
                        </span>
                        <span className="text">Finetuned Models</span>
                      </Link>
                    </li> */}
                  </ul>
                </div>

                <div className="nav_group">
                  <h2 className="group__title">User Tools</h2>
                  <ul className="group__list">
                    <li>
                      <Link
                        href="/ai"
                        className="fn__tooltip menu__item"
                        data-position="right"
                        title="Image Generation"
                      >
                        <span className="icon">
                          <img src="svg/image.svg" alt="" className="fn__svg" />
                        </span>
                        <span className="text">Text</span>
                      </Link>
                    </li>
                    <li>
                      <Link
                        href="/image"
                        className="fn__tooltip menu__item"
                        data-position="right"
                        title="AI Chat Bot"
                      >
                        <span className="icon">
                          <img src="svg/chat.svg" alt="" className="fn__svg" />
                        </span>
                        <span className="text">Image</span>
                      </Link>
                    </li>
                    <li>
                      <Link
                        href="/ai"
                        className="fn__tooltip menu__item"
                        data-position="right"
                        title="AI Chat Bot"
                      >
                        <span className="icon">
                          <img src="svg/chat.svg" alt="" className="fn__svg" />
                        </span>
                        <span className="text">Audio</span>
                      </Link>
                    </li>
                    <li>
                      <Link
                        href="/ai"
                        className="fn__tooltip menu__item"
                        data-position="right"
                        title="AI Chat Bot"
                      >
                        <span className="icon">
                          <img src="svg/chat.svg" alt="" className="fn__svg" />
                        </span>
                        <span className="text">Video</span>
                      </Link>
                    </li>
                  </ul>
                </div>

                <div className="nav_group">
                  <h2 className="group__title">Support</h2>
                  <ul className="group__list">
                    {/* <li>
                      <Link
                        href="/"
                        className="fn__tooltip menu__item"
                        data-position="right"
                        title="Pricing"
                      >
                        <span className="icon">
                          <img
                            src="svg/dollar.svg"
                            alt=""
                            className="fn__svg"
                          />
                        </span>
                        <span className="text">Pricing</span>
                      </Link>
                    </li>
                    <li className="menu-item-has-children">
                      <Link
                        href="/"
                        className="fn__tooltip menu__item"
                        title="FAQ &amp; Help"
                        data-position="right"
                      >
                        <span className="icon">
                          <img
                            src="svg/question.svg"
                            alt=""
                            className="fn__svg"
                          />
                        </span>
                        <span className="text">FAQ &amp; Help</span>
                        <span className="trigger">
                          <img src="svg/arrow.svg" alt="" className="fn__svg" />
                        </span>
                      </Link>
                      <ul className="sub-menu">
                        <li>
                          <Link href="/">
                            <span className="text">Documentation</span>
                          </Link>
                        </li>
                        <li>
                          <Link href="/">
                            <span className="text">FAQ</span>
                          </Link>
                        </li>
                        <li>
                          <Link href="/">
                            <span className="text">
                              Changelog<span className="fn__sup">(4.1.2)</span>
                            </span>
                          </Link>
                        </li>
                        <li>
                          <Link href="/">
                            <span className="text">Contact Us</span>
                          </Link>
                        </li>
                      </ul>
                    </li> */}
                    <li>
                      <Link
                        href="/"
                        onClick={handleLogout}
                        className="fn__tooltip menu__item"
                        data-position="right"
                        title="Log Out"
                      >
                        <span className="icon">
                          <img
                            src="svg/logout.svg"
                            alt=""
                            className="fn__svg"
                          />
                        </span>
                        <span className="text">Log Out</span>
                      </Link>
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="techwave_fn_content">
              <div className="techwave_fn_page">
                <div className="techwave_fn_home">
                  <div className="section_home">
                    <div className="section_left">
                      <div className="techwave_fn_title_holder">
                        <h1 className="title">
                          Unleash Your Creativity with AI
                        </h1>
                        <p className="desc">
                          Generate your ideas into stunning visuals
                        </p>
                      </div>

                      <div className="techwave_fn_interactive_list">
                        <ul>
                          <li>
                            <div className="item">
                              <Link
                                href="/ai"
                                onClick={() => {
                                  route.push("/ai");
                                }}
                              >
                                <span className="icon">
                                  <img
                                    src="svg/image.svg"
                                    alt=""
                                    className="fn__svg"
                                  />
                                </span>
                                <h2 className="title">Text</h2>
                                <p className="desc">
                                  Files that include textual data. For now only
                                  Pdfs and text files, in future we will add all
                                  the necessary formats of text files
                                </p>
                                <span className="arrow">
                                  <img
                                    src="svg/arrow.svg"
                                    alt=""
                                    className="fn__svg"
                                  />
                                </span>
                              </Link>
                            </div>
                          </li>
                          <li>
                            <div className="item">
                              <Link
                                href="/image"
                                onClick={() => {
                                  route.push("/image");
                                }}
                              >
                                <span className="icon">
                                  <img
                                    src="svg/image.svg"
                                    alt=""
                                    className="fn__svg"
                                  />
                                </span>
                                <h2 className="title">Image</h2>
                                <p className="desc">
                                  We process the image and give information in
                                  the four domains, image captioning, entity
                                  recognition, counting objects & Image based
                                  QnA
                                </p>
                                <span className="arrow">
                                  <img
                                    src="svg/arrow.svg"
                                    alt=""
                                    className="fn__svg"
                                  />
                                </span>
                              </Link>
                            </div>
                          </li>
                          <li>
                            <div className="item">
                              <Link
                                href="/ai"
                                onClick={() => {
                                  route.push("/ai");
                                }}
                              >
                                <span className="icon">
                                  <img
                                    src="svg/chat.svg"
                                    alt=""
                                    className="fn__svg"
                                  />
                                </span>
                                <h2 className="title">Audio</h2>
                                <p className="desc">
                                  An AI chatbot, short for artificial
                                  intelligence chatbot, is a computer program
                                  designed to simulate human-like conversations
                                  and provide automated responses to user
                                  queries or prompts.{" "}
                                </p>
                                <span className="arrow">
                                  <img
                                    src="svg/arrow.svg"
                                    alt=""
                                    className="fn__svg"
                                  />
                                </span>
                              </Link>
                            </div>
                          </li>
                          <li>
                            <div className="item">
                              <Link
                                href="/ai"
                                onClick={() => {
                                  route.push("/ai");
                                }}
                              >
                                <span className="icon">
                                  <img
                                    src="svg/chat.svg"
                                    alt=""
                                    className="fn__svg"
                                  />
                                </span>
                                <h2 className="title">Video</h2>
                                <p className="desc">
                                  Input are the Video files (urls for now) which
                                  are then processed by model and provide its
                                  text based summarisation and also QnA on video
                                </p>
                                <span className="arrow">
                                  <img
                                    src="svg/arrow.svg"
                                    alt=""
                                    className="fn__svg"
                                  />
                                </span>
                              </Link>
                            </div>
                          </li>
                        </ul>
                      </div>
                    </div>
                    <div className="section_right">
                      <div className="company_info">
                        <img src="images/CCPD_proj_prev_ui.png" alt="" />
                        <p className="fn__animated_text">
                          The official server of TECH-AI, a text-to-image AI
                          where your imagination is the only limit. We’re
                          building market-leading features that will give you
                          greater control over your generations.
                        </p>
                        <hr />
                        <div className="fn__members">
                          <div className="active item">
                            <span className="circle"></span>
                            <span className="text">1,154,694 Online</span>
                          </div>
                          <div className="item">
                            <span className="circle"></span>
                            <span className="text">77,345,912 Members</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <footer className="techwave_fn_footer">
                <div className="techwave_fn_footer_content">
                  <div className="copyright">
                    <p>2023© ITB23</p>
                  </div>
                  <div className="menu_items">
                    <ul>
                      <li>
                        <Link href="/">Terms of Service</Link>
                      </li>
                      <li>
                        <Link href="/">Privacy Policy</Link>
                      </li>
                    </ul>
                  </div>
                </div>
              </footer>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
