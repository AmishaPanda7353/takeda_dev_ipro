/* eslint-disable prefer-template */
import { chatResponse } from "components/ConversationalFlow/types";

const constants = {
  organizationName: "Tiger Analytics",
};

export const fxCode = '';

export const ENDPOINTS = {
  chatDetails: "api/chat_initiation" + fxCode,
  question: "api/chat_question" + fxCode,
  generateReport: "api/report_generate" + fxCode,
  resetChat: "api/chat_reset" + fxCode,
  newTopic: "api/chat_context" + fxCode,
  chatHistory: "api/chat_history" + fxCode,
  chatFeedback: "api/chat_feedback" + fxCode,
  dataNames: "api/get_data_names" + fxCode,
  preTrack: "api/pre_track" + fxCode,
  track1: "api/text_to_query" + fxCode,
  track2: "api/query_to_chart" + fxCode,
  track3: "api/table_to_insights" + fxCode,
  postTrack: "api/post_track" + fxCode
};

export const CHATRESPONSE: chatResponse = {
  user_id: "",
  report_list: [
    {
      "report_id": 1,
      "report_name": "Report 1",
      "report_url": "https://s3://path/to/report1",
      chat_id: 11454,
    },
    {
      "report_id": 2,
      "report_name": "Report 2",
      "report_url": "https://s3://path/to/report2",
      chat_id: 214752,
    }
  ],
  chat_details:
  {
    "chat_id": 2,
    "user_id": "2",
    context_id: 2,
    "data_name": 'Finance',
    "chat_name": "chat 9HF9VO",
    "is_active": true,
    chat_history: {
      1: [
        {
          "question_id": 1235,
          context_id: 1,
          "feedback": true,
          "question": "Which shipment are delayed0?",
          "answer_id": 102,
          "type": "clarification",
          "data": {
            "content": "I did not understand can you give me more business context"
          },
          "created_time": "2023-06-01 12:48:27"
        },
      ],
      2: [
        {
          "question_id": 1,
          context_id: 2,
          "feedback": false,
          "question": "What is the status of shipment at Location 140?",
          "answer_id": 101,
          "type": "insight",
          "data": [
            {
              "insight_type": "sql_query",
              "content": "SELECT COUNT(DISTINCT sto_sap_invoice) AS shipment_count FROM invoice_data WHERE source_location_name LIKE '%Location 14%'"
            },
            {
              "insight_type": "chart",
              "content": {
                "data": [
                  {
                    "mode": "lines",
                    "x": [
                      42,
                      43,
                      44
                    ],
                    "y": [
                      12,
                      25,
                      1
                    ],
                    "type": "scatter"
                  }
                ],
                "layout": {
                  "template": {
                    "data": {
                      "histogram2dcontour": [
                        {
                          "type": "histogram2dcontour",
                          "colorbar": {
                            "outlinewidth": 0,
                            "ticks": ""
                          },
                          "colorscale": [
                            [
                              0.0,
                              "#0d0887"
                            ],
                            [
                              0.1111111111111111,
                              "#46039f"
                            ],
                            [
                              0.2222222222222222,
                              "#7201a8"
                            ],
                            [
                              0.3333333333333333,
                              "#9c179e"
                            ],
                            [
                              0.4444444444444444,
                              "#bd3786"
                            ],
                            [
                              0.5555555555555556,
                              "#d8576b"
                            ],
                            [
                              0.6666666666666666,
                              "#ed7953"
                            ],
                            [
                              0.7777777777777778,
                              "#fb9f3a"
                            ],
                            [
                              0.8888888888888888,
                              "#fdca26"
                            ],
                            [
                              1.0,
                              "#f0f921"
                            ]
                          ]
                        }
                      ],
                      "choropleth": [
                        {
                          "type": "choropleth",
                          "colorbar": {
                            "outlinewidth": 0,
                            "ticks": ""
                          }
                        }
                      ],
                      "histogram2d": [
                        {
                          "type": "histogram2d",
                          "colorbar": {
                            "outlinewidth": 0,
                            "ticks": ""
                          },
                          "colorscale": [
                            [
                              0.0,
                              "#0d0887"
                            ],
                            [
                              0.1111111111111111,
                              "#46039f"
                            ],
                            [
                              0.2222222222222222,
                              "#7201a8"
                            ],
                            [
                              0.3333333333333333,
                              "#9c179e"
                            ],
                            [
                              0.4444444444444444,
                              "#bd3786"
                            ],
                            [
                              0.5555555555555556,
                              "#d8576b"
                            ],
                            [
                              0.6666666666666666,
                              "#ed7953"
                            ],
                            [
                              0.7777777777777778,
                              "#fb9f3a"
                            ],
                            [
                              0.8888888888888888,
                              "#fdca26"
                            ],
                            [
                              1.0,
                              "#f0f921"
                            ]
                          ]
                        }
                      ],
                      "heatmap": [
                        {
                          "type": "heatmap",
                          "colorbar": {
                            "outlinewidth": 0,
                            "ticks": ""
                          },
                          "colorscale": [
                            [
                              0.0,
                              "#0d0887"
                            ],
                            [
                              0.1111111111111111,
                              "#46039f"
                            ],
                            [
                              0.2222222222222222,
                              "#7201a8"
                            ],
                            [
                              0.3333333333333333,
                              "#9c179e"
                            ],
                            [
                              0.4444444444444444,
                              "#bd3786"
                            ],
                            [
                              0.5555555555555556,
                              "#d8576b"
                            ],
                            [
                              0.6666666666666666,
                              "#ed7953"
                            ],
                            [
                              0.7777777777777778,
                              "#fb9f3a"
                            ],
                            [
                              0.8888888888888888,
                              "#fdca26"
                            ],
                            [
                              1.0,
                              "#f0f921"
                            ]
                          ]
                        }
                      ],
                      "heatmapgl": [
                        {
                          "type": "heatmapgl",
                          "colorbar": {
                            "outlinewidth": 0,
                            "ticks": ""
                          },
                          "colorscale": [
                            [
                              0.0,
                              "#0d0887"
                            ],
                            [
                              0.1111111111111111,
                              "#46039f"
                            ],
                            [
                              0.2222222222222222,
                              "#7201a8"
                            ],
                            [
                              0.3333333333333333,
                              "#9c179e"
                            ],
                            [
                              0.4444444444444444,
                              "#bd3786"
                            ],
                            [
                              0.5555555555555556,
                              "#d8576b"
                            ],
                            [
                              0.6666666666666666,
                              "#ed7953"
                            ],
                            [
                              0.7777777777777778,
                              "#fb9f3a"
                            ],
                            [
                              0.8888888888888888,
                              "#fdca26"
                            ],
                            [
                              1.0,
                              "#f0f921"
                            ]
                          ]
                        }
                      ],
                      "contourcarpet": [
                        {
                          "type": "contourcarpet",
                          "colorbar": {
                            "outlinewidth": 0,
                            "ticks": ""
                          }
                        }
                      ],
                      "contour": [
                        {
                          "type": "contour",
                          "colorbar": {
                            "outlinewidth": 0,
                            "ticks": ""
                          },
                          "colorscale": [
                            [
                              0.0,
                              "#0d0887"
                            ],
                            [
                              0.1111111111111111,
                              "#46039f"
                            ],
                            [
                              0.2222222222222222,
                              "#7201a8"
                            ],
                            [
                              0.3333333333333333,
                              "#9c179e"
                            ],
                            [
                              0.4444444444444444,
                              "#bd3786"
                            ],
                            [
                              0.5555555555555556,
                              "#d8576b"
                            ],
                            [
                              0.6666666666666666,
                              "#ed7953"
                            ],
                            [
                              0.7777777777777778,
                              "#fb9f3a"
                            ],
                            [
                              0.8888888888888888,
                              "#fdca26"
                            ],
                            [
                              1.0,
                              "#f0f921"
                            ]
                          ]
                        }
                      ],
                      "surface": [
                        {
                          "type": "surface",
                          "colorbar": {
                            "outlinewidth": 0,
                            "ticks": ""
                          },
                          "colorscale": [
                            [
                              0.0,
                              "#0d0887"
                            ],
                            [
                              0.1111111111111111,
                              "#46039f"
                            ],
                            [
                              0.2222222222222222,
                              "#7201a8"
                            ],
                            [
                              0.3333333333333333,
                              "#9c179e"
                            ],
                            [
                              0.4444444444444444,
                              "#bd3786"
                            ],
                            [
                              0.5555555555555556,
                              "#d8576b"
                            ],
                            [
                              0.6666666666666666,
                              "#ed7953"
                            ],
                            [
                              0.7777777777777778,
                              "#fb9f3a"
                            ],
                            [
                              0.8888888888888888,
                              "#fdca26"
                            ],
                            [
                              1.0,
                              "#f0f921"
                            ]
                          ]
                        }
                      ],
                      "mesh3d": [
                        {
                          "type": "mesh3d",
                          "colorbar": {
                            "outlinewidth": 0,
                            "ticks": ""
                          }
                        }
                      ],
                      "scatter": [
                        {
                          "fillpattern": {
                            "fillmode": "overlay",
                            "size": 10,
                            "solidity": 0.2
                          },
                          "type": "scatter"
                        }
                      ],
                      "parcoords": [
                        {
                          "type": "parcoords",
                          "line": {
                            "colorbar": {
                              "outlinewidth": 0,
                              "ticks": ""
                            }
                          }
                        }
                      ],
                      "scatterpolargl": [
                        {
                          "type": "scatterpolargl",
                          "marker": {
                            "colorbar": {
                              "outlinewidth": 0,
                              "ticks": ""
                            }
                          }
                        }
                      ],
                      "bar": [
                        {
                          "error_x": {
                            "color": "#2a3f5f"
                          },
                          "error_y": {
                            "color": "#2a3f5f"
                          },
                          "marker": {
                            "line": {
                              "color": "#E5ECF6",
                              "width": 0.5
                            },
                            "pattern": {
                              "fillmode": "overlay",
                              "size": 10,
                              "solidity": 0.2
                            }
                          },
                          "type": "bar"
                        }
                      ],
                      "scattergeo": [
                        {
                          "type": "scattergeo",
                          "marker": {
                            "colorbar": {
                              "outlinewidth": 0,
                              "ticks": ""
                            }
                          }
                        }
                      ],
                      "scatterpolar": [
                        {
                          "type": "scatterpolar",
                          "marker": {
                            "colorbar": {
                              "outlinewidth": 0,
                              "ticks": ""
                            }
                          }
                        }
                      ],
                      "histogram": [
                        {
                          "marker": {
                            "pattern": {
                              "fillmode": "overlay",
                              "size": 10,
                              "solidity": 0.2
                            }
                          },
                          "type": "histogram"
                        }
                      ],
                      "scattergl": [
                        {
                          "type": "scattergl",
                          "marker": {
                            "colorbar": {
                              "outlinewidth": 0,
                              "ticks": ""
                            }
                          }
                        }
                      ],
                      "scatter3d": [
                        {
                          "type": "scatter3d",
                          "line": {
                            "colorbar": {
                              "outlinewidth": 0,
                              "ticks": ""
                            }
                          },
                          "marker": {
                            "colorbar": {
                              "outlinewidth": 0,
                              "ticks": ""
                            }
                          }
                        }
                      ],
                      "scattermapbox": [
                        {
                          "type": "scattermapbox",
                          "marker": {
                            "colorbar": {
                              "outlinewidth": 0,
                              "ticks": ""
                            }
                          }
                        }
                      ],
                      "scatterternary": [
                        {
                          "type": "scatterternary",
                          "marker": {
                            "colorbar": {
                              "outlinewidth": 0,
                              "ticks": ""
                            }
                          }
                        }
                      ],
                      "scattercarpet": [
                        {
                          "type": "scattercarpet",
                          "marker": {
                            "colorbar": {
                              "outlinewidth": 0,
                              "ticks": ""
                            }
                          }
                        }
                      ],
                      "carpet": [
                        {
                          "aaxis": {
                            "endlinecolor": "#2a3f5f",
                            "gridcolor": "white",
                            "linecolor": "white",
                            "minorgridcolor": "white",
                            "startlinecolor": "#2a3f5f"
                          },
                          "baxis": {
                            "endlinecolor": "#2a3f5f",
                            "gridcolor": "white",
                            "linecolor": "white",
                            "minorgridcolor": "white",
                            "startlinecolor": "#2a3f5f"
                          },
                          "type": "carpet"
                        }
                      ],
                      "table": [
                        {
                          "cells": {
                            "fill": {
                              "color": "#EBF0F8"
                            },
                            "line": {
                              "color": "white"
                            }
                          },
                          "header": {
                            "fill": {
                              "color": "#C8D4E3"
                            },
                            "line": {
                              "color": "white"
                            }
                          },
                          "type": "table"
                        }
                      ],
                      "barpolar": [
                        {
                          "marker": {
                            "line": {
                              "color": "#E5ECF6",
                              "width": 0.5
                            },
                            "pattern": {
                              "fillmode": "overlay",
                              "size": 10,
                              "solidity": 0.2
                            }
                          },
                          "type": "barpolar"
                        }
                      ],
                      "pie": [
                        {
                          "automargin": true,
                          "type": "pie"
                        }
                      ]
                    },
                    "layout": {
                      "autotypenumbers": "strict",
                      "colorway": [
                        "#636efa",
                        "#EF553B",
                        "#00cc96",
                        "#ab63fa",
                        "#FFA15A",
                        "#19d3f3",
                        "#FF6692",
                        "#B6E880",
                        "#FF97FF",
                        "#FECB52"
                      ],
                      "font": {
                        "color": "#2a3f5f"
                      },
                      "hovermode": "closest",
                      "hoverlabel": {
                        "align": "left"
                      },
                      "paper_bgcolor": "white",
                      "plot_bgcolor": "#E5ECF6",
                      "polar": {
                        "bgcolor": "#E5ECF6",
                        "angularaxis": {
                          "gridcolor": "white",
                          "linecolor": "white",
                          "ticks": ""
                        },
                        "radialaxis": {
                          "gridcolor": "white",
                          "linecolor": "white",
                          "ticks": ""
                        }
                      },
                      "ternary": {
                        "bgcolor": "#E5ECF6",
                        "aaxis": {
                          "gridcolor": "white",
                          "linecolor": "white",
                          "ticks": ""
                        },
                        "baxis": {
                          "gridcolor": "white",
                          "linecolor": "white",
                          "ticks": ""
                        },
                        "caxis": {
                          "gridcolor": "white",
                          "linecolor": "white",
                          "ticks": ""
                        }
                      },
                      "coloraxis": {
                        "colorbar": {
                          "outlinewidth": 0,
                          "ticks": ""
                        }
                      },
                      "colorscale": {
                        "sequential": [
                          [
                            0.0,
                            "#0d0887"
                          ],
                          [
                            0.1111111111111111,
                            "#46039f"
                          ],
                          [
                            0.2222222222222222,
                            "#7201a8"
                          ],
                          [
                            0.3333333333333333,
                            "#9c179e"
                          ],
                          [
                            0.4444444444444444,
                            "#bd3786"
                          ],
                          [
                            0.5555555555555556,
                            "#d8576b"
                          ],
                          [
                            0.6666666666666666,
                            "#ed7953"
                          ],
                          [
                            0.7777777777777778,
                            "#fb9f3a"
                          ],
                          [
                            0.8888888888888888,
                            "#fdca26"
                          ],
                          [
                            1.0,
                            "#f0f921"
                          ]
                        ],
                        "sequentialminus": [
                          [
                            0.0,
                            "#0d0887"
                          ],
                          [
                            0.1111111111111111,
                            "#46039f"
                          ],
                          [
                            0.2222222222222222,
                            "#7201a8"
                          ],
                          [
                            0.3333333333333333,
                            "#9c179e"
                          ],
                          [
                            0.4444444444444444,
                            "#bd3786"
                          ],
                          [
                            0.5555555555555556,
                            "#d8576b"
                          ],
                          [
                            0.6666666666666666,
                            "#ed7953"
                          ],
                          [
                            0.7777777777777778,
                            "#fb9f3a"
                          ],
                          [
                            0.8888888888888888,
                            "#fdca26"
                          ],
                          [
                            1.0,
                            "#f0f921"
                          ]
                        ],
                        "diverging": [
                          [
                            0,
                            "#8e0152"
                          ],
                          [
                            0.1,
                            "#c51b7d"
                          ],
                          [
                            0.2,
                            "#de77ae"
                          ],
                          [
                            0.3,
                            "#f1b6da"
                          ],
                          [
                            0.4,
                            "#fde0ef"
                          ],
                          [
                            0.5,
                            "#f7f7f7"
                          ],
                          [
                            0.6,
                            "#e6f5d0"
                          ],
                          [
                            0.7,
                            "#b8e186"
                          ],
                          [
                            0.8,
                            "#7fbc41"
                          ],
                          [
                            0.9,
                            "#4d9221"
                          ],
                          [
                            1,
                            "#276419"
                          ]
                        ]
                      },
                      "xaxis": {
                        "gridcolor": "white",
                        "linecolor": "white",
                        "ticks": "",
                        "title": {
                          "standoff": 15
                        },
                        "zerolinecolor": "white",
                        "automargin": true,
                        "zerolinewidth": 2
                      },
                      "yaxis": {
                        "gridcolor": "white",
                        "linecolor": "white",
                        "ticks": "",
                        "title": {
                          "standoff": 15
                        },
                        "zerolinecolor": "white",
                        "automargin": true,
                        "zerolinewidth": 2
                      },
                      "scene": {
                        "xaxis": {
                          "backgroundcolor": "#E5ECF6",
                          "gridcolor": "white",
                          "linecolor": "white",
                          "showbackground": true,
                          "ticks": "",
                          "zerolinecolor": "white",
                          "gridwidth": 2
                        },
                        "yaxis": {
                          "backgroundcolor": "#E5ECF6",
                          "gridcolor": "white",
                          "linecolor": "white",
                          "showbackground": true,
                          "ticks": "",
                          "zerolinecolor": "white",
                          "gridwidth": 2
                        },
                        "zaxis": {
                          "backgroundcolor": "#E5ECF6",
                          "gridcolor": "white",
                          "linecolor": "white",
                          "showbackground": true,
                          "ticks": "",
                          "zerolinecolor": "white",
                          "gridwidth": 2
                        }
                      },
                      "shapedefaults": {
                        "line": {
                          "color": "#2a3f5f"
                        }
                      },
                      "annotationdefaults": {
                        "arrowcolor": "#2a3f5f",
                        "arrowhead": 0,
                        "arrowwidth": 1
                      },
                      "geo": {
                        "bgcolor": "white",
                        "landcolor": "#E5ECF6",
                        "subunitcolor": "white",
                        "showland": true,
                        "showlakes": true,
                        "lakecolor": "white"
                      },
                      "title": {
                        "x": 0.05
                      },
                      "mapbox": {
                        "style": "light"
                      }
                    }
                  },
                  "title": {
                    "text": "Shipments Delivered Each Week in October-2022 at Location 100"
                  },
                  "xaxis": {
                    "title": {
                      "text": "Week Number"
                    }
                  },
                  "yaxis": {
                    "title": {
                      "text": "Shipment Count"
                    }
                  }
                }
              }
            },
            {
              "insight_type": "summary",
              "content": "The table contains data for three brands: BoldBites, FlavorCrave, and RidgeBite, with ordered_qty values of 44,879,360, 37,633,471, and 24,690,799 respectively, and all having an unshipped_qty of 4. - There is a strong positive correlation between brand popularity and ordered_qty, with BoldBites being the most popular and RidgeBite being the least popular."
            }
          ],
          "created_time": "2023-06-01 12:36:46"
        },
        {
          "question_id": 1235,
          context_id: 2,
          "feedback": false,
          "question": "Which shipment are delayed1412?",
          "answer_id": 102,
          "type": "clarification",
          "data": {
            "content": "I did not understand can you give me more business context"
          },
          "created_time": "2023-06-01 12:48:27"
        },
      ],
      3: [
        {
          "question_id": 2235,
          context_id: 3,
          "feedback": false,
          "question": "Where is table data434?",
          "answer_id": 105,
          "type": "table_selector",
          "data": {
            "content": [[{ "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
            { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
            { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
            { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
            { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" }
            ], [{ "firstCol": "Data", "secondCol": "Data", "thirdCol": 10, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
            { "firstCol": "Data", "secondCol": "Data", "thirdCol": 10, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
            { "firstCol": "Data", "secondCol": "Data", "thirdCol": 10, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
            { "firstCol": "Data", "secondCol": "Data", "thirdCol": 10, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
            { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" }
            ], [{ "firstCol": "Data", "secondCol": "Data", "thirdCol": 20, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
            { "firstCol": "Data", "secondCol": "Data", "thirdCol": 20, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
            { "firstCol": "Data", "secondCol": "Data", "thirdCol": 20, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
            { "firstCol": "Data", "secondCol": "Data", "thirdCol": 20, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
            { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" }
            ]]
          },
          "created_time": "2023-06-01 12:59:39"
        }
      ],
      4: [
        {
          "question_id": 1,
          context_id: 4,
          "feedback": false,
          "question": "What is the status of shipment at Location 142243?",
          "answer_id": 101,
          "type": "insight",
          "data": [
            {
              "insight_type": "sql_query",
              "content": "SELECT COUNT(DISTINCT sto_sap_invoice) AS shipment_count FROM invoice_data WHERE source_location_name LIKE '%Location 14%'"
            },
            {
              "insight_type": "table_selector",
              "content": [[{ "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
              { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
              { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
              { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
              { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" }
              ], [{ "firstCol": "Data", "secondCol": "Data", "thirdCol": 10, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
              { "firstCol": "Data", "secondCol": "Data", "thirdCol": 10, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
              { "firstCol": "Data", "secondCol": "Data", "thirdCol": 10, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
              { "firstCol": "Data", "secondCol": "Data", "thirdCol": 10, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
              { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" }
              ], [{ "firstCol": "Data", "secondCol": "Data", "thirdCol": 20, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
              { "firstCol": "Data", "secondCol": "Data", "thirdCol": 20, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
              { "firstCol": "Data", "secondCol": "Data", "thirdCol": 20, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
              { "firstCol": "Data", "secondCol": "Data", "thirdCol": 20, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" },
              { "firstCol": "Data", "secondCol": "Data", "thirdCol": 50, "fourthCol": "Data", "fifthCol": 50, "sixthCol": "Data", "seventhCol": "Data" }
              ]]
            },
            {
              "insight_type": "summary",
              "content": "The table contains data for three brands: BoldBites, FlavorCrave, and RidgeBite, with ordered_qty values of 44,879,360, 37,633,471, and 24,690,799 respectively, and all having an unshipped_qty of 4. - There is a strong positive correlation between brand popularity and ordered_qty, with BoldBites being the most popular and RidgeBite being the least popular."
            }
          ],
          "created_time": "2023-06-01 12:36:46"
        },
        {
          "question_id": 1235,
          context_id: 4,
          "feedback": true,
          "question": "Which shipment are delayed123454?",
          "answer_id": 102,
          "type": "clarification",
          "data": {
            "content": "I did not understand can you give me more business context"
          },
          "created_time": "2023-06-01 12:48:27"
        },
        {
          "question_id": 1,
          context_id: 4,
          "feedback": true,
          "question": "What is the status of shipment at Location 141453?",
          "answer_id": 101,
          "type": "insight",
          "data": [
            {
              "insight_type": "number",
              "content": "33"
            },
            {
              "insight_type": "sql_query",
              "content": "SELECT COUNT(DISTINCT sto_sap_invoice) AS shipment_count FROM invoice_data WHERE source_location_name LIKE '%Location 14%'"
            },
            {
              "insight_type": "summary",
              "content": "The table contains data for three brands: BoldBites, FlavorCrave, and RidgeBite, with ordered_qty values of 44,879,360, 37,633,471, and 24,690,799 respectively, and all having an unshipped_qty of 4. - There is a strong positive correlation between brand popularity and ordered_qty, with BoldBites being the most popular and RidgeBite being the least popular."
            }
          ],
          "created_time": "2023-06-01 12:36:46"
        },

      ],
      5: [
        {
          "question_id": 1235,
          context_id: 1,
          "feedback": false,
          "question": "Which shipment are delayed0?",
          "answer_id": 102,
          "type": "clarification",
          "data": {
            "content": "I did not understand can you give me more business context"
          },
          "created_time": "2023-06-01 12:48:27"
        },
      ],
    }
  }
}

export const track1Response = {
  error: null,
  user_id: "rishabh.chourasi@tigeranalytics.com",
  question_id: 6364,
  answer_id: 6406,
  category: "insights",
  content: {
    error: "",
    status: ["success"],
    type: "insights",
    data: [
      {
        insight_type: "sql_query",
        content:
          "WITH avg_on_time AS (\n    SELECT carrier_name, AVG(CASE WHEN current_state = 'on-time' THEN 1 ELSE 0 END) AS avg_on_time\n    FROM invoice_data\n    GROUP BY carrier_name\n), \navg_delayed AS (\n    SELECT carrier_name, AVG(CASE WHEN current_state = 'delayed' THEN 1 ELSE 0 END) AS avg_delayed\n    FROM invoice_data\n    GROUP BY carrier_name\n)\nSELECT aot.carrier_name, aot.avg_on_time, ad.avg_delayed\nFROM avg_on_time aot\nJOIN avg_delayed ad ON aot.carrier_name = ad.carrier_name\nWHERE aot.avg_on_time > (SELECT AVG(CASE WHEN current_state = 'on-time' THEN 1 ELSE 0 END) FROM invoice_data)\nAND ad.avg_delayed > (SELECT AVG(CASE WHEN current_state = 'delayed' THEN 1 ELSE 0 END) FROM invoice_data)",
        table: [
          '[{"carrier_name":"Carrier 10","avg_on_time":0.31,"avg_delayed":0.14},{"carrier_name":"Carrier 16","avg_on_time":0.26,"avg_delayed":0.1},{"carrier_name":"Carrier 20","avg_on_time":0.42,"avg_delayed":0.19},{"carrier_name":"Carrier 24","avg_on_time":0.48,"avg_delayed":0.14},{"carrier_name":"Carrier 26","avg_on_time":0.49,"avg_delayed":0.17},{"carrier_name":"Carrier 28","avg_on_time":0.22,"avg_delayed":0.11},{"carrier_name":"Carrier 30","avg_on_time":0.4,"avg_delayed":0.27}]',
        ],
        error: "",
        showError: false,
        bot_response: "",
      },
    ],
    created_time: "2024-06-03_07_16_46_079512Z",
    completion_response: "",
    response_for_history: "",
    question_index: "Q_20240530121648875541_1_4355",
    output_path:
      "/data/supply_chain_management/output_folder_test/fastapi_k8s_v2/Q_20240530121648875541_1_4355",
    engines: {
      TEXT_TO_QUERY: "gpt_35_turbo",
      FOLLOWUP_QUESTION_TAGGING: "gpt_35_turbo",
      QUERY_TO_CHART_TYPE: "gpt_35_turbo",
      QUERY_TO_CHART_CODE: "gpt_35_turbo",
      TABLE_TO_INSIGHT_QUESTIONS: "gpt_test",
      QUESTIONS_TO_INSIGHTS: "gpt_test",
      SUMMARIZE_INSIGHTS: "gpt_test",
      SUMMARIZE_TABLES: "gpt_test",
      INSIGHT_QUESTIONS_TO_CODE: "gpt_test",
    },
  },
};

export const track2Response = {
  error: null,
  user_id: "rishabh.chourasi@tigeranalytics.com",
  question_id: 6364,
  answer_id: 6407,
  category: "insights",
  content: {
    error: "",
    status: ["success", "success"],
    type: "insights",
    data: [
      {
        insight_type: "sql_query",
        content:
          "WITH avg_on_time AS (\n    SELECT carrier_name, AVG(CASE WHEN current_state = 'on-time' THEN 1 ELSE 0 END) AS avg_on_time\n    FROM invoice_data\n    GROUP BY carrier_name\n), \navg_delayed AS (\n    SELECT carrier_name, AVG(CASE WHEN current_state = 'delayed' THEN 1 ELSE 0 END) AS avg_delayed\n    FROM invoice_data\n    GROUP BY carrier_name\n)\nSELECT aot.carrier_name, aot.avg_on_time, ad.avg_delayed\nFROM avg_on_time aot\nJOIN avg_delayed ad ON aot.carrier_name = ad.carrier_name\nWHERE aot.avg_on_time > (SELECT AVG(CASE WHEN current_state = 'on-time' THEN 1 ELSE 0 END) FROM invoice_data)\nAND ad.avg_delayed > (SELECT AVG(CASE WHEN current_state = 'delayed' THEN 1 ELSE 0 END) FROM invoice_data)",
        error: "",
        showError: false,
        bot_response: "",
      },
      {
        insight_type: "chart",
        content: [
          {
            data: [
              {
                marker: {
                  color: "green",
                },
                name: "Above Average",
                x: [
                  "Carrier 10",
                  "Carrier 16",
                  "Carrier 20",
                  "Carrier 24",
                  "Carrier 26",
                  "Carrier 28",
                  "Carrier 30",
                ],
                y: [0.42, 0.48, 0.49, 0.4],
                type: "bar",
              },
              {
                marker: {
                  color: "red",
                },
                name: "Below Average",
                x: [
                  "Carrier 10",
                  "Carrier 16",
                  "Carrier 20",
                  "Carrier 24",
                  "Carrier 26",
                  "Carrier 28",
                  "Carrier 30",
                ],
                y: [0.31, 0.26, 0.22],
                type: "bar",
              },
            ],
            layout: {
              template: {
                data: {
                  barpolar: [
                    {
                      marker: {
                        line: {
                          color: "#E5ECF6",
                          width: 0.5,
                        },
                        pattern: {
                          fillmode: "overlay",
                          size: 10,
                          solidity: 0.2,
                        },
                      },
                      type: "barpolar",
                    },
                  ],
                  bar: [
                    {
                      error_x: {
                        color: "#2a3f5f",
                      },
                      error_y: {
                        color: "#2a3f5f",
                      },
                      marker: {
                        line: {
                          color: "#E5ECF6",
                          width: 0.5,
                        },
                        pattern: {
                          fillmode: "overlay",
                          size: 10,
                          solidity: 0.2,
                        },
                      },
                      type: "bar",
                    },
                  ],
                  carpet: [
                    {
                      aaxis: {
                        endlinecolor: "#2a3f5f",
                        gridcolor: "white",
                        linecolor: "white",
                        minorgridcolor: "white",
                        startlinecolor: "#2a3f5f",
                      },
                      baxis: {
                        endlinecolor: "#2a3f5f",
                        gridcolor: "white",
                        linecolor: "white",
                        minorgridcolor: "white",
                        startlinecolor: "#2a3f5f",
                      },
                      type: "carpet",
                    },
                  ],
                  choropleth: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      type: "choropleth",
                    },
                  ],
                  contourcarpet: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      type: "contourcarpet",
                    },
                  ],
                  contour: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      colorscale: [
                        [0, "#0d0887"],
                        [0.1111111111111111, "#46039f"],
                        [0.2222222222222222, "#7201a8"],
                        [0.3333333333333333, "#9c179e"],
                        [0.4444444444444444, "#bd3786"],
                        [0.5555555555555556, "#d8576b"],
                        [0.6666666666666666, "#ed7953"],
                        [0.7777777777777778, "#fb9f3a"],
                        [0.8888888888888888, "#fdca26"],
                        [1, "#f0f921"],
                      ],
                      type: "contour",
                    },
                  ],
                  heatmapgl: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      colorscale: [
                        [0, "#0d0887"],
                        [0.1111111111111111, "#46039f"],
                        [0.2222222222222222, "#7201a8"],
                        [0.3333333333333333, "#9c179e"],
                        [0.4444444444444444, "#bd3786"],
                        [0.5555555555555556, "#d8576b"],
                        [0.6666666666666666, "#ed7953"],
                        [0.7777777777777778, "#fb9f3a"],
                        [0.8888888888888888, "#fdca26"],
                        [1, "#f0f921"],
                      ],
                      type: "heatmapgl",
                    },
                  ],
                  heatmap: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      colorscale: [
                        [0, "#0d0887"],
                        [0.1111111111111111, "#46039f"],
                        [0.2222222222222222, "#7201a8"],
                        [0.3333333333333333, "#9c179e"],
                        [0.4444444444444444, "#bd3786"],
                        [0.5555555555555556, "#d8576b"],
                        [0.6666666666666666, "#ed7953"],
                        [0.7777777777777778, "#fb9f3a"],
                        [0.8888888888888888, "#fdca26"],
                        [1, "#f0f921"],
                      ],
                      type: "heatmap",
                    },
                  ],
                  histogram2dcontour: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      colorscale: [
                        [0, "#0d0887"],
                        [0.1111111111111111, "#46039f"],
                        [0.2222222222222222, "#7201a8"],
                        [0.3333333333333333, "#9c179e"],
                        [0.4444444444444444, "#bd3786"],
                        [0.5555555555555556, "#d8576b"],
                        [0.6666666666666666, "#ed7953"],
                        [0.7777777777777778, "#fb9f3a"],
                        [0.8888888888888888, "#fdca26"],
                        [1, "#f0f921"],
                      ],
                      type: "histogram2dcontour",
                    },
                  ],
                  histogram2d: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      colorscale: [
                        [0, "#0d0887"],
                        [0.1111111111111111, "#46039f"],
                        [0.2222222222222222, "#7201a8"],
                        [0.3333333333333333, "#9c179e"],
                        [0.4444444444444444, "#bd3786"],
                        [0.5555555555555556, "#d8576b"],
                        [0.6666666666666666, "#ed7953"],
                        [0.7777777777777778, "#fb9f3a"],
                        [0.8888888888888888, "#fdca26"],
                        [1, "#f0f921"],
                      ],
                      type: "histogram2d",
                    },
                  ],
                  histogram: [
                    {
                      marker: {
                        pattern: {
                          fillmode: "overlay",
                          size: 10,
                          solidity: 0.2,
                        },
                      },
                      type: "histogram",
                    },
                  ],
                  mesh3d: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      type: "mesh3d",
                    },
                  ],
                  parcoords: [
                    {
                      line: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "parcoords",
                    },
                  ],
                  pie: [
                    {
                      automargin: true,
                      type: "pie",
                    },
                  ],
                  scatter3d: [
                    {
                      line: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scatter3d",
                    },
                  ],
                  scattercarpet: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scattercarpet",
                    },
                  ],
                  scattergeo: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scattergeo",
                    },
                  ],
                  scattergl: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scattergl",
                    },
                  ],
                  scattermapbox: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scattermapbox",
                    },
                  ],
                  scatterpolargl: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scatterpolargl",
                    },
                  ],
                  scatterpolar: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scatterpolar",
                    },
                  ],
                  scatter: [
                    {
                      fillpattern: {
                        fillmode: "overlay",
                        size: 10,
                        solidity: 0.2,
                      },
                      type: "scatter",
                    },
                  ],
                  scatterternary: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scatterternary",
                    },
                  ],
                  surface: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      colorscale: [
                        [0, "#0d0887"],
                        [0.1111111111111111, "#46039f"],
                        [0.2222222222222222, "#7201a8"],
                        [0.3333333333333333, "#9c179e"],
                        [0.4444444444444444, "#bd3786"],
                        [0.5555555555555556, "#d8576b"],
                        [0.6666666666666666, "#ed7953"],
                        [0.7777777777777778, "#fb9f3a"],
                        [0.8888888888888888, "#fdca26"],
                        [1, "#f0f921"],
                      ],
                      type: "surface",
                    },
                  ],
                  table: [
                    {
                      cells: {
                        fill: {
                          color: "#EBF0F8",
                        },
                        line: {
                          color: "white",
                        },
                      },
                      header: {
                        fill: {
                          color: "#C8D4E3",
                        },
                        line: {
                          color: "white",
                        },
                      },
                      type: "table",
                    },
                  ],
                },
                layout: {
                  annotationdefaults: {
                    arrowcolor: "#2a3f5f",
                    arrowhead: 0,
                    arrowwidth: 1,
                  },
                  autotypenumbers: "strict",
                  coloraxis: {
                    colorbar: {
                      outlinewidth: 0,
                      ticks: "",
                    },
                  },
                  colorscale: {
                    diverging: [
                      [0, "#8e0152"],
                      [0.1, "#c51b7d"],
                      [0.2, "#de77ae"],
                      [0.3, "#f1b6da"],
                      [0.4, "#fde0ef"],
                      [0.5, "#f7f7f7"],
                      [0.6, "#e6f5d0"],
                      [0.7, "#b8e186"],
                      [0.8, "#7fbc41"],
                      [0.9, "#4d9221"],
                      [1, "#276419"],
                    ],
                    sequential: [
                      [0, "#0d0887"],
                      [0.1111111111111111, "#46039f"],
                      [0.2222222222222222, "#7201a8"],
                      [0.3333333333333333, "#9c179e"],
                      [0.4444444444444444, "#bd3786"],
                      [0.5555555555555556, "#d8576b"],
                      [0.6666666666666666, "#ed7953"],
                      [0.7777777777777778, "#fb9f3a"],
                      [0.8888888888888888, "#fdca26"],
                      [1, "#f0f921"],
                    ],
                    sequentialminus: [
                      [0, "#0d0887"],
                      [0.1111111111111111, "#46039f"],
                      [0.2222222222222222, "#7201a8"],
                      [0.3333333333333333, "#9c179e"],
                      [0.4444444444444444, "#bd3786"],
                      [0.5555555555555556, "#d8576b"],
                      [0.6666666666666666, "#ed7953"],
                      [0.7777777777777778, "#fb9f3a"],
                      [0.8888888888888888, "#fdca26"],
                      [1, "#f0f921"],
                    ],
                  },
                  colorway: [
                    "#636efa",
                    "#EF553B",
                    "#00cc96",
                    "#ab63fa",
                    "#FFA15A",
                    "#19d3f3",
                    "#FF6692",
                    "#B6E880",
                    "#FF97FF",
                    "#FECB52",
                  ],
                  font: {
                    color: "#2a3f5f",
                  },
                  geo: {
                    bgcolor: "white",
                    lakecolor: "white",
                    landcolor: "#E5ECF6",
                    showlakes: true,
                    showland: true,
                    subunitcolor: "white",
                  },
                  hoverlabel: {
                    align: "left",
                  },
                  hovermode: "closest",
                  mapbox: {
                    style: "light",
                  },
                  paper_bgcolor: "white",
                  plot_bgcolor: "#E5ECF6",
                  polar: {
                    angularaxis: {
                      gridcolor: "white",
                      linecolor: "white",
                      ticks: "",
                    },
                    bgcolor: "#E5ECF6",
                    radialaxis: {
                      gridcolor: "white",
                      linecolor: "white",
                      ticks: "",
                    },
                  },
                  scene: {
                    xaxis: {
                      backgroundcolor: "#E5ECF6",
                      gridcolor: "white",
                      gridwidth: 2,
                      linecolor: "white",
                      showbackground: true,
                      ticks: "",
                      zerolinecolor: "white",
                    },
                    yaxis: {
                      backgroundcolor: "#E5ECF6",
                      gridcolor: "white",
                      gridwidth: 2,
                      linecolor: "white",
                      showbackground: true,
                      ticks: "",
                      zerolinecolor: "white",
                    },
                    zaxis: {
                      backgroundcolor: "#E5ECF6",
                      gridcolor: "white",
                      gridwidth: 2,
                      linecolor: "white",
                      showbackground: true,
                      ticks: "",
                      zerolinecolor: "white",
                    },
                  },
                  shapedefaults: {
                    line: {
                      color: "#2a3f5f",
                    },
                  },
                  ternary: {
                    aaxis: {
                      gridcolor: "white",
                      linecolor: "white",
                      ticks: "",
                    },
                    baxis: {
                      gridcolor: "white",
                      linecolor: "white",
                      ticks: "",
                    },
                    bgcolor: "#E5ECF6",
                    caxis: {
                      gridcolor: "white",
                      linecolor: "white",
                      ticks: "",
                    },
                  },
                  title: {
                    x: 0.05,
                  },
                  xaxis: {
                    automargin: true,
                    gridcolor: "white",
                    linecolor: "white",
                    ticks: "",
                    title: {
                      standoff: 15,
                    },
                    zerolinecolor: "white",
                    zerolinewidth: 2,
                  },
                  yaxis: {
                    automargin: true,
                    gridcolor: "white",
                    linecolor: "white",
                    ticks: "",
                    title: {
                      standoff: 15,
                    },
                    zerolinecolor: "white",
                    zerolinewidth: 2,
                  },
                },
              },
              title: {
                text: "On-time Shipments by Carrier",
              },
              xaxis: {
                title: {
                  text: "Carrier Name",
                },
              },
              yaxis: {
                title: {
                  text: "Average On-time Shipments",
                },
              },
            },
          },
        ],
        error: "",
        showError: false,
        table: [
          '[{"carrier_name":"Carrier 10","avg_on_time":0.31,"avg_delayed":0.14},{"carrier_name":"Carrier 16","avg_on_time":0.26,"avg_delayed":0.1},{"carrier_name":"Carrier 20","avg_on_time":0.42,"avg_delayed":0.19},{"carrier_name":"Carrier 24","avg_on_time":0.48,"avg_delayed":0.14},{"carrier_name":"Carrier 26","avg_on_time":0.49,"avg_delayed":0.17},{"carrier_name":"Carrier 28","avg_on_time":0.22,"avg_delayed":0.11},{"carrier_name":"Carrier 30","avg_on_time":0.4,"avg_delayed":0.27}]',
        ],
        bot_response: "",
      },
    ],
    created_time: "2024-05-31_09_40_05_564106Z",
    completion_response: "",
    response_for_history: "",
    question_index: "Q_20240530121648875541_1_4355",
    output_path:
      "/data/supply_chain_management/output_folder_test/fastapi_k8s_v2/Q_20240530121648875541_1_4355",
    engines: {
      TEXT_TO_QUERY: "gpt_35_turbo",
      FOLLOWUP_QUESTION_TAGGING: "gpt_35_turbo",
      QUERY_TO_CHART_TYPE: "gpt_35_turbo",
      QUERY_TO_CHART_CODE: "gpt_35_turbo",
      TABLE_TO_INSIGHT_QUESTIONS: "gpt_test",
      QUESTIONS_TO_INSIGHTS: "gpt_test",
      SUMMARIZE_INSIGHTS: "gpt_test",
      SUMMARIZE_TABLES: "gpt_test",
      INSIGHT_QUESTIONS_TO_CODE: "gpt_test",
    },
  },
};

export const track3Response = {
  error: null,
  user_id: "rishabh.chourasi@tigeranalytics.com",
  question_id: 6364,
  answer_id: 6408,
  category: "insights",
  content: {
    error: "",
    status: ["success", "success", "success"],
    type: "insights",
    data: [
      {
        insight_type: "sql_query",
        content:
          "WITH avg_on_time AS (\n    SELECT carrier_name, AVG(CASE WHEN current_state = 'on-time' THEN 1 ELSE 0 END) AS avg_on_time\n    FROM invoice_data\n    GROUP BY carrier_name\n), \navg_delayed AS (\n    SELECT carrier_name, AVG(CASE WHEN current_state = 'delayed' THEN 1 ELSE 0 END) AS avg_delayed\n    FROM invoice_data\n    GROUP BY carrier_name\n)\nSELECT aot.carrier_name, aot.avg_on_time, ad.avg_delayed\nFROM avg_on_time aot\nJOIN avg_delayed ad ON aot.carrier_name = ad.carrier_name\nWHERE aot.avg_on_time > (SELECT AVG(CASE WHEN current_state = 'on-time' THEN 1 ELSE 0 END) FROM invoice_data)\nAND ad.avg_delayed > (SELECT AVG(CASE WHEN current_state = 'delayed' THEN 1 ELSE 0 END) FROM invoice_data)",
        error: "",
        showError: false,
        bot_response: "",
      },
      {
        insight_type: "chart",
        content: [
          {
            data: [
              {
                marker: {
                  color: "green",
                },
                name: "Above Average",
                x: [
                  "Carrier 10",
                  "Carrier 16",
                  "Carrier 20",
                  "Carrier 24",
                  "Carrier 26",
                  "Carrier 28",
                  "Carrier 30",
                ],
                y: [0.42, 0.48, 0.49, 0.4],
                type: "bar",
              },
              {
                marker: {
                  color: "red",
                },
                name: "Below Average",
                x: [
                  "Carrier 10",
                  "Carrier 16",
                  "Carrier 20",
                  "Carrier 24",
                  "Carrier 26",
                  "Carrier 28",
                  "Carrier 30",
                ],
                y: [0.31, 0.26, 0.22],
                type: "bar",
              },
            ],
            layout: {
              template: {
                data: {
                  barpolar: [
                    {
                      marker: {
                        line: {
                          color: "#E5ECF6",
                          width: 0.5,
                        },
                        pattern: {
                          fillmode: "overlay",
                          size: 10,
                          solidity: 0.2,
                        },
                      },
                      type: "barpolar",
                    },
                  ],
                  bar: [
                    {
                      error_x: {
                        color: "#2a3f5f",
                      },
                      error_y: {
                        color: "#2a3f5f",
                      },
                      marker: {
                        line: {
                          color: "#E5ECF6",
                          width: 0.5,
                        },
                        pattern: {
                          fillmode: "overlay",
                          size: 10,
                          solidity: 0.2,
                        },
                      },
                      type: "bar",
                    },
                  ],
                  carpet: [
                    {
                      aaxis: {
                        endlinecolor: "#2a3f5f",
                        gridcolor: "white",
                        linecolor: "white",
                        minorgridcolor: "white",
                        startlinecolor: "#2a3f5f",
                      },
                      baxis: {
                        endlinecolor: "#2a3f5f",
                        gridcolor: "white",
                        linecolor: "white",
                        minorgridcolor: "white",
                        startlinecolor: "#2a3f5f",
                      },
                      type: "carpet",
                    },
                  ],
                  choropleth: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      type: "choropleth",
                    },
                  ],
                  contourcarpet: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      type: "contourcarpet",
                    },
                  ],
                  contour: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      colorscale: [
                        [0, "#0d0887"],
                        [0.1111111111111111, "#46039f"],
                        [0.2222222222222222, "#7201a8"],
                        [0.3333333333333333, "#9c179e"],
                        [0.4444444444444444, "#bd3786"],
                        [0.5555555555555556, "#d8576b"],
                        [0.6666666666666666, "#ed7953"],
                        [0.7777777777777778, "#fb9f3a"],
                        [0.8888888888888888, "#fdca26"],
                        [1, "#f0f921"],
                      ],
                      type: "contour",
                    },
                  ],
                  heatmapgl: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      colorscale: [
                        [0, "#0d0887"],
                        [0.1111111111111111, "#46039f"],
                        [0.2222222222222222, "#7201a8"],
                        [0.3333333333333333, "#9c179e"],
                        [0.4444444444444444, "#bd3786"],
                        [0.5555555555555556, "#d8576b"],
                        [0.6666666666666666, "#ed7953"],
                        [0.7777777777777778, "#fb9f3a"],
                        [0.8888888888888888, "#fdca26"],
                        [1, "#f0f921"],
                      ],
                      type: "heatmapgl",
                    },
                  ],
                  heatmap: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      colorscale: [
                        [0, "#0d0887"],
                        [0.1111111111111111, "#46039f"],
                        [0.2222222222222222, "#7201a8"],
                        [0.3333333333333333, "#9c179e"],
                        [0.4444444444444444, "#bd3786"],
                        [0.5555555555555556, "#d8576b"],
                        [0.6666666666666666, "#ed7953"],
                        [0.7777777777777778, "#fb9f3a"],
                        [0.8888888888888888, "#fdca26"],
                        [1, "#f0f921"],
                      ],
                      type: "heatmap",
                    },
                  ],
                  histogram2dcontour: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      colorscale: [
                        [0, "#0d0887"],
                        [0.1111111111111111, "#46039f"],
                        [0.2222222222222222, "#7201a8"],
                        [0.3333333333333333, "#9c179e"],
                        [0.4444444444444444, "#bd3786"],
                        [0.5555555555555556, "#d8576b"],
                        [0.6666666666666666, "#ed7953"],
                        [0.7777777777777778, "#fb9f3a"],
                        [0.8888888888888888, "#fdca26"],
                        [1, "#f0f921"],
                      ],
                      type: "histogram2dcontour",
                    },
                  ],
                  histogram2d: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      colorscale: [
                        [0, "#0d0887"],
                        [0.1111111111111111, "#46039f"],
                        [0.2222222222222222, "#7201a8"],
                        [0.3333333333333333, "#9c179e"],
                        [0.4444444444444444, "#bd3786"],
                        [0.5555555555555556, "#d8576b"],
                        [0.6666666666666666, "#ed7953"],
                        [0.7777777777777778, "#fb9f3a"],
                        [0.8888888888888888, "#fdca26"],
                        [1, "#f0f921"],
                      ],
                      type: "histogram2d",
                    },
                  ],
                  histogram: [
                    {
                      marker: {
                        pattern: {
                          fillmode: "overlay",
                          size: 10,
                          solidity: 0.2,
                        },
                      },
                      type: "histogram",
                    },
                  ],
                  mesh3d: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      type: "mesh3d",
                    },
                  ],
                  parcoords: [
                    {
                      line: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "parcoords",
                    },
                  ],
                  pie: [
                    {
                      automargin: true,
                      type: "pie",
                    },
                  ],
                  scatter3d: [
                    {
                      line: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scatter3d",
                    },
                  ],
                  scattercarpet: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scattercarpet",
                    },
                  ],
                  scattergeo: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scattergeo",
                    },
                  ],
                  scattergl: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scattergl",
                    },
                  ],
                  scattermapbox: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scattermapbox",
                    },
                  ],
                  scatterpolargl: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scatterpolargl",
                    },
                  ],
                  scatterpolar: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scatterpolar",
                    },
                  ],
                  scatter: [
                    {
                      fillpattern: {
                        fillmode: "overlay",
                        size: 10,
                        solidity: 0.2,
                      },
                      type: "scatter",
                    },
                  ],
                  scatterternary: [
                    {
                      marker: {
                        colorbar: {
                          outlinewidth: 0,
                          ticks: "",
                        },
                      },
                      type: "scatterternary",
                    },
                  ],
                  surface: [
                    {
                      colorbar: {
                        outlinewidth: 0,
                        ticks: "",
                      },
                      colorscale: [
                        [0, "#0d0887"],
                        [0.1111111111111111, "#46039f"],
                        [0.2222222222222222, "#7201a8"],
                        [0.3333333333333333, "#9c179e"],
                        [0.4444444444444444, "#bd3786"],
                        [0.5555555555555556, "#d8576b"],
                        [0.6666666666666666, "#ed7953"],
                        [0.7777777777777778, "#fb9f3a"],
                        [0.8888888888888888, "#fdca26"],
                        [1, "#f0f921"],
                      ],
                      type: "surface",
                    },
                  ],
                  table: [
                    {
                      cells: {
                        fill: {
                          color: "#EBF0F8",
                        },
                        line: {
                          color: "white",
                        },
                      },
                      header: {
                        fill: {
                          color: "#C8D4E3",
                        },
                        line: {
                          color: "white",
                        },
                      },
                      type: "table",
                    },
                  ],
                },
                layout: {
                  annotationdefaults: {
                    arrowcolor: "#2a3f5f",
                    arrowhead: 0,
                    arrowwidth: 1,
                  },
                  autotypenumbers: "strict",
                  coloraxis: {
                    colorbar: {
                      outlinewidth: 0,
                      ticks: "",
                    },
                  },
                  colorscale: {
                    diverging: [
                      [0, "#8e0152"],
                      [0.1, "#c51b7d"],
                      [0.2, "#de77ae"],
                      [0.3, "#f1b6da"],
                      [0.4, "#fde0ef"],
                      [0.5, "#f7f7f7"],
                      [0.6, "#e6f5d0"],
                      [0.7, "#b8e186"],
                      [0.8, "#7fbc41"],
                      [0.9, "#4d9221"],
                      [1, "#276419"],
                    ],
                    sequential: [
                      [0, "#0d0887"],
                      [0.1111111111111111, "#46039f"],
                      [0.2222222222222222, "#7201a8"],
                      [0.3333333333333333, "#9c179e"],
                      [0.4444444444444444, "#bd3786"],
                      [0.5555555555555556, "#d8576b"],
                      [0.6666666666666666, "#ed7953"],
                      [0.7777777777777778, "#fb9f3a"],
                      [0.8888888888888888, "#fdca26"],
                      [1, "#f0f921"],
                    ],
                    sequentialminus: [
                      [0, "#0d0887"],
                      [0.1111111111111111, "#46039f"],
                      [0.2222222222222222, "#7201a8"],
                      [0.3333333333333333, "#9c179e"],
                      [0.4444444444444444, "#bd3786"],
                      [0.5555555555555556, "#d8576b"],
                      [0.6666666666666666, "#ed7953"],
                      [0.7777777777777778, "#fb9f3a"],
                      [0.8888888888888888, "#fdca26"],
                      [1, "#f0f921"],
                    ],
                  },
                  colorway: [
                    "#636efa",
                    "#EF553B",
                    "#00cc96",
                    "#ab63fa",
                    "#FFA15A",
                    "#19d3f3",
                    "#FF6692",
                    "#B6E880",
                    "#FF97FF",
                    "#FECB52",
                  ],
                  font: {
                    color: "#2a3f5f",
                  },
                  geo: {
                    bgcolor: "white",
                    lakecolor: "white",
                    landcolor: "#E5ECF6",
                    showlakes: true,
                    showland: true,
                    subunitcolor: "white",
                  },
                  hoverlabel: {
                    align: "left",
                  },
                  hovermode: "closest",
                  mapbox: {
                    style: "light",
                  },
                  paper_bgcolor: "white",
                  plot_bgcolor: "#E5ECF6",
                  polar: {
                    angularaxis: {
                      gridcolor: "white",
                      linecolor: "white",
                      ticks: "",
                    },
                    bgcolor: "#E5ECF6",
                    radialaxis: {
                      gridcolor: "white",
                      linecolor: "white",
                      ticks: "",
                    },
                  },
                  scene: {
                    xaxis: {
                      backgroundcolor: "#E5ECF6",
                      gridcolor: "white",
                      gridwidth: 2,
                      linecolor: "white",
                      showbackground: true,
                      ticks: "",
                      zerolinecolor: "white",
                    },
                    yaxis: {
                      backgroundcolor: "#E5ECF6",
                      gridcolor: "white",
                      gridwidth: 2,
                      linecolor: "white",
                      showbackground: true,
                      ticks: "",
                      zerolinecolor: "white",
                    },
                    zaxis: {
                      backgroundcolor: "#E5ECF6",
                      gridcolor: "white",
                      gridwidth: 2,
                      linecolor: "white",
                      showbackground: true,
                      ticks: "",
                      zerolinecolor: "white",
                    },
                  },
                  shapedefaults: {
                    line: {
                      color: "#2a3f5f",
                    },
                  },
                  ternary: {
                    aaxis: {
                      gridcolor: "white",
                      linecolor: "white",
                      ticks: "",
                    },
                    baxis: {
                      gridcolor: "white",
                      linecolor: "white",
                      ticks: "",
                    },
                    bgcolor: "#E5ECF6",
                    caxis: {
                      gridcolor: "white",
                      linecolor: "white",
                      ticks: "",
                    },
                  },
                  title: {
                    x: 0.05,
                  },
                  xaxis: {
                    automargin: true,
                    gridcolor: "white",
                    linecolor: "white",
                    ticks: "",
                    title: {
                      standoff: 15,
                    },
                    zerolinecolor: "white",
                    zerolinewidth: 2,
                  },
                  yaxis: {
                    automargin: true,
                    gridcolor: "white",
                    linecolor: "white",
                    ticks: "",
                    title: {
                      standoff: 15,
                    },
                    zerolinecolor: "white",
                    zerolinewidth: 2,
                  },
                },
              },
              title: {
                text: "On-time Shipments by Carrier",
              },
              xaxis: {
                title: {
                  text: "Carrier Name",
                },
              },
              yaxis: {
                title: {
                  text: "Average On-time Shipments",
                },
              },
            },
          },
        ],
        error: "",
        showError: false,
        table: [
          '[{"carrier_name":"Carrier 10","avg_on_time":0.31,"avg_delayed":0.14},{"carrier_name":"Carrier 16","avg_on_time":0.26,"avg_delayed":0.1},{"carrier_name":"Carrier 20","avg_on_time":0.42,"avg_delayed":0.19},{"carrier_name":"Carrier 24","avg_on_time":0.48,"avg_delayed":0.14},{"carrier_name":"Carrier 26","avg_on_time":0.49,"avg_delayed":0.17},{"carrier_name":"Carrier 28","avg_on_time":0.22,"avg_delayed":0.11},{"carrier_name":"Carrier 30","avg_on_time":0.4,"avg_delayed":0.27}]',
        ],
        bot_response: "",
      },
      {
        insight_type: "summary",
        content:
          "- The carrier with the highest average on-time shipments is Carrier 26. \n- When compared to Carrier 26, other carriers have lower on-time shipment averages, with Carrier 28 having the largest difference of -0.27.\n- The average delay in shipments varies across carriers, with Carrier 30 having the highest average delay of 0.27. This data can be used to improve delivery efficiency by identifying and addressing issues with carriers that have high average delays.\n- Carriers 20, 24, 26, and 30 have above average on-time shipments, while Carriers 10, 16, and 28 have below average on-time shipments.",
        error: "",
        showError: false,
        bot_response: "",
      },
    ],
    created_time: "2024-05-31_09_40_05_564106Z",
    completion_response: "Done and done! Here are the results you asked for.",
    response_for_history: "Done and done! Here are the results you asked for.",
    question_index: "Q_20240530121648875541_1_4355",
    output_path:
      "/data/supply_chain_management/output_folder_test/fastapi_k8s_v2/Q_20240530121648875541_1_4355",
    engines: {
      TEXT_TO_QUERY: "gpt_35_turbo",
      FOLLOWUP_QUESTION_TAGGING: "gpt_35_turbo",
      QUERY_TO_CHART_TYPE: "gpt_35_turbo",
      QUERY_TO_CHART_CODE: "gpt_35_turbo",
      TABLE_TO_INSIGHT_QUESTIONS: "gpt_test",
      QUESTIONS_TO_INSIGHTS: "gpt_test",
      SUMMARIZE_INSIGHTS: "gpt_test",
      SUMMARIZE_TABLES: "gpt_test",
      INSIGHT_QUESTIONS_TO_CODE: "gpt_test",
    },
  },
};
export default constants;
